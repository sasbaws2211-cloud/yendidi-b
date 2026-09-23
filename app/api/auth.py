import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID
import secrets
import uuid

import pyotp
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlmodel import select

from app.core.config import settings
from app.core.dependencies import get_current_user, get_session, security
from app.core.email import send_email
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_token,
    verify_password,
)
from app.db.models import (
    OneTimePassword,
    OTPPurpose,
    Parent,
    RefreshToken,
    School,
    Student,
    User,
    Vendor,
)
from app.db.session import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LoginRequest(BaseModel):
    identifier: str
    password: str
    device_id: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class OtpRequest(BaseModel):
    identifier: str
    purpose: Optional[OTPPurpose] = OTPPurpose.login


class OtpVerifyRequest(BaseModel):
    identifier: str
    code: str
    device_id: str


class MfaVerifyRequest(BaseModel):
    code: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int


class AccessTokenResponse(BaseModel):
    access_token: str
    expires_in: int


class UserResponse(BaseModel):
    id: UUID
    role: str
    full_name: str
    email: Optional[str]
    phone: Optional[str]
    school_id: Optional[UUID]
    profile_id: Optional[UUID] = None
    # Student-only. A student cannot read their own roster row — /schools/{id}
    # and /schools/{id}/students both 403 for that role — so this is the only
    # place these can reach the student's own screen. See docs/BACKEND_HANDOVER.md §5.
    student_code: Optional[str] = None
    class_name: Optional[str] = None
    level: Optional[str] = None
    allergies: Optional[List[str]] = None
    dietary_notes: Optional[str] = None
    school_name: Optional[str] = None

    class Config:
        from_attributes = True


def _build_access_token_data(user: User, device_id: str, profile_id: Optional[str] = None, mfa_verified: bool = False) -> str:
    return create_access_token(
        subject=str(user.id),
        role=user.role,
        device_id=device_id,
        school_id=str(user.school_id) if user.school_id else None,
        profile_id=profile_id,
        mfa_verified=mfa_verified,
    )


async def _get_profile_id(user: User, session: AsyncSession) -> Optional[str]:
    if user.role == "student":
        student_stmt = select(Student).where(Student.user_id == user.id)
        student_result = await session.execute(student_stmt)
        student = student_result.scalar_one_or_none()
        return str(student.id) if student else None
    if user.role == "parent":
        parent_stmt = select(Parent).where(Parent.user_id == user.id)
        parent_result = await session.execute(parent_stmt)
        parent = parent_result.scalar_one_or_none()
        return str(parent.id) if parent else None
    if user.role == "vendor":
        vendor_stmt = select(Vendor).where(Vendor.user_id == user.id)
        vendor_result = await session.execute(vendor_stmt)
        vendor = vendor_result.scalar_one_or_none()
        return str(vendor.id) if vendor else None
    return None


async def _send_otp(user: User, code: str, purpose: OTPPurpose) -> None:
    if not user.email:
        if settings.debug_log_otp_on_delivery_failure:
            logger.warning("OTP for %s (%s): %s — no email on file, not delivered", user.id, purpose, code)
            return
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No email available for OTP delivery")
    subject = "Your Y3ndidi login code"
    body = f"Your verification code is {code}. It expires in 10 minutes."
    try:
        await send_email(user.email, subject, body, body)
    except Exception:
        if not settings.debug_log_otp_on_delivery_failure:
            raise
        # Only reached with the flag on, which defaults off and is meant for a
        # local instance with no working email provider — never trust this
        # branch to be unreachable in a real deployment.
        logger.warning("OTP for %s (%s): %s — email delivery failed, logged instead", user.id, purpose, code)


@router.post("/otp/request")
async def request_otp(request: OtpRequest, session: AsyncSession = Depends(get_session)):
    statement = select(User).where((User.email == request.identifier) | (User.phone == request.identifier))
    result = await session.execute(statement)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    code = f"{secrets.randbelow(900000) + 100000}"
    otp = OneTimePassword(
        id=uuid.uuid4(),
        user_id=user.id,
        recipient=user.email or request.identifier,
        purpose=request.purpose,
        code_hash=hash_token(code),
        expires_at=_utc_now() + timedelta(minutes=10),
        created_at=_utc_now(),
    )
    session.add(otp)
    await session.commit()
    await _send_otp(user, code, request.purpose)
    return {"status": "pending", "message": "OTP sent"}


@router.post("/otp/verify", response_model=TokenResponse)
async def verify_otp(request: OtpVerifyRequest, session: AsyncSession = Depends(get_session)):
    statement = select(User).where((User.email == request.identifier) | (User.phone == request.identifier))
    result = await session.execute(statement)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    otp_stmt = (
        select(OneTimePassword)
        .where(
            OneTimePassword.user_id == user.id,
            OneTimePassword.used_at.is_(None),
            OneTimePassword.expires_at > _utc_now(),
        )
        .order_by(OneTimePassword.created_at.desc())
    )
    otp_result = await session.execute(otp_stmt)
    otp = otp_result.scalar_one_or_none()
    if not otp or hash_token(request.code) != otp.code_hash:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid OTP code")

    otp.used_at = _utc_now()
    session.add(otp)
    await session.commit()

    profile_id = await _get_profile_id(user, session)
    mfa_verified = not (user.role in {"school_admin", "super_admin"} and bool(user.mfa_secret))
    access_token = _build_access_token_data(user, request.device_id, profile_id=profile_id, mfa_verified=mfa_verified)
    refresh_plain = create_refresh_token()
    refresh = RefreshToken(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=hash_token(refresh_plain),
        device_id=request.device_id,
        expires_at=_utc_now() + timedelta(days=settings.refresh_token_expire_days),
        created_at=_utc_now(),
    )
    session.add(refresh)
    await session.commit()

    return TokenResponse(access_token=access_token, refresh_token=refresh_plain, expires_in=settings.access_token_expire_minutes * 60)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, session: AsyncSession = Depends(get_session)):
    statement = select(User).where((User.email == request.identifier) | (User.phone == request.identifier))
    result = await session.execute(statement)
    user = result.scalar_one_or_none()
    if not user or not user.password_hash or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect identifier or password")

    profile_id = await _get_profile_id(user, session)
    mfa_verified = not (user.role in {"school_admin", "super_admin"} and bool(user.mfa_secret))
    access_token = _build_access_token_data(user, request.device_id, profile_id=profile_id, mfa_verified=mfa_verified)
    refresh_plain = create_refresh_token()
    refresh = RefreshToken(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=hash_token(refresh_plain),
        device_id=request.device_id,
        expires_at=_utc_now() + timedelta(days=settings.refresh_token_expire_days),
        created_at=_utc_now(),
    )
    session.add(refresh)
    await session.commit()

    return TokenResponse(access_token=access_token, refresh_token=refresh_plain, expires_in=settings.access_token_expire_minutes * 60)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest, session: AsyncSession = Depends(get_session)):
    hash_value = hash_token(request.refresh_token)
    statement = select(RefreshToken).where(RefreshToken.token_hash == hash_value)
    result = await session.execute(statement)
    refresh = result.scalar_one_or_none()
    if not refresh or refresh.revoked_at is not None or refresh.expires_at < _utc_now():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user_stmt = select(User).where(User.id == refresh.user_id)
    user_result = await session.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    new_refresh_plain = create_refresh_token()
    replacement_id = uuid.uuid4()
    session.add(
        RefreshToken(
            id=replacement_id,
            user_id=user.id,
            token_hash=hash_token(new_refresh_plain),
            device_id=refresh.device_id,
            expires_at=_utc_now() + timedelta(days=settings.refresh_token_expire_days),
            created_at=_utc_now(),
        )
    )
    # Flushed before the old row is revoked: `replaced_by` is a plain FK column
    # with no ORM relationship tying the two rows together, so the unit of work
    # has no dependency to order by and can flush this update before the insert
    # it points at — which asyncpg then rejects outright.
    await session.flush()
    refresh.revoked_at = _utc_now()
    refresh.replaced_by = replacement_id
    session.add(refresh)
    await session.commit()

    profile_id = await _get_profile_id(user, session)
    mfa_verified = not (user.role in {"school_admin", "super_admin"} and bool(user.mfa_secret))
    access_token = create_access_token(
        subject=str(user.id),
        role=user.role,
        device_id=refresh.device_id,
        school_id=str(user.school_id) if user.school_id else None,
        profile_id=profile_id,
        mfa_verified=mfa_verified,
    )
    return TokenResponse(access_token=access_token, refresh_token=new_refresh_plain, expires_in=settings.access_token_expire_minutes * 60)


@router.post("/logout")
async def logout(request: LogoutRequest, session: AsyncSession = Depends(get_session)):
    hash_value = hash_token(request.refresh_token)
    statement = select(RefreshToken).where(RefreshToken.token_hash == hash_value)
    result = await session.execute(statement)
    refresh = result.scalar_one_or_none()
    if refresh and refresh.revoked_at is None:
        refresh.revoked_at = _utc_now()
        session.add(refresh)
        await session.commit()
    return {"status": "ok"}


@router.post("/mfa/verify", response_model=AccessTokenResponse)
async def verify_mfa(
    request: MfaVerifyRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
):
    token_data = decode_access_token(credentials.credentials)
    if token_data.role not in {"school_admin", "super_admin"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MFA verification is only required for admin roles")
    user_stmt = select(User).where(User.id == token_data.sub)
    user_result = await session.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    if not user or not user.mfa_secret:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MFA is not configured for this account")

    totp = pyotp.TOTP(user.mfa_secret)
    if not totp.verify(request.code, valid_window=1):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MFA token")

    access_token = create_access_token(
        subject=str(user.id),
        role=user.role,
        device_id=token_data.device_id,
        school_id=token_data.school_id,
        profile_id=token_data.profile_id,
        mfa_verified=True,
    )
    return AccessTokenResponse(access_token=access_token, expires_in=settings.access_token_expire_minutes * 60)


@router.get("/me", response_model=UserResponse)
async def me(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    profile_id = await _get_profile_id(current_user, session)
    response = UserResponse.from_orm(current_user)
    response.profile_id = UUID(profile_id) if profile_id else None

    if current_user.role == "student":
        student_stmt = select(Student).where(Student.user_id == current_user.id)
        student_result = await session.execute(student_stmt)
        student = student_result.scalar_one_or_none()
        if student:
            response.student_code = student.student_code
            response.class_name = student.class_name
            response.level = student.level
            response.allergies = student.allergies
            response.dietary_notes = student.dietary_notes
            if student.school_id:
                school_stmt = select(School).where(School.id == student.school_id)
                school_result = await session.execute(school_stmt)
                school = school_result.scalar_one_or_none()
                if school:
                    response.school_name = school.name

    return response
