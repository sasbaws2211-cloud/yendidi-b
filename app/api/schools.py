import csv
import io
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlmodel import select

from app.core.dependencies import get_current_user, get_session
from app.db.models import (
    Announcement,
    AuditLog,
    FaceTemplate,
    Guardianship,
    School,
    SchoolStatus,
    Student,
    Subscription,
    SubscriptionPlan,
    SubscriptionStatus,
    User,
    Role,
    Wallet,
)
from app.db.session import AsyncSession

router = APIRouter()


class SchoolResponse(BaseModel):
    id: UUID
    name: str
    code: str
    region: str
    district: str
    address: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    headteacher: Optional[str]
    levels: List[str]
    status: SchoolStatus

    class Config:
        from_attributes = True


class StudentCreateRequest(BaseModel):
    student_code: str
    first_name: str
    last_name: str
    class_name: str
    level: str
    allergies: Optional[List[str]] = []
    dietary_notes: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class StudentUpdateRequest(BaseModel):
    # Partial update — every field must default to None, or Pydantic v2
    # requires it present on every PATCH regardless of `exclude_unset`.
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    class_name: Optional[str] = None
    level: Optional[str] = None
    allergies: Optional[List[str]] = None
    dietary_notes: Optional[str] = None


class SchoolCreateRequest(BaseModel):
    name: str
    code: str
    region: str
    district: str
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    headteacher: Optional[str] = None
    levels: Optional[List[str]] = []


class SchoolStatusUpdateRequest(BaseModel):
    status: SchoolStatus


class SchoolStudentResponse(BaseModel):
    id: UUID
    student_code: str
    first_name: str
    last_name: str
    class_name: str
    level: str
    allergies: List[str]
    dietary_notes: Optional[str]
    guardian_ids: List[UUID] = []

    class Config:
        from_attributes = True


class SchoolEnrollmentResult(BaseModel):
    imported: int
    skipped: int
    errors: List[str]


# Expected embedding length per supported model version, so a mismatched or
# garbage vector is rejected at enrollment rather than silently stored and
# discovered broken the first time a kiosk tries to compare against it. A
# kiosk trusts `model_version` to mean "these numbers are comparable this
# way" — enrollment is the one place that guarantee has to be enforced.
SUPPORTED_FACE_MODELS = {
    "arcface-r100-v1": 512,
    "virtuoturing-v1": 512,
}


class FaceEnrollmentRequest(BaseModel):
    # The embedding a trusted device computed from a live capture — never a
    # photo. Nothing on this request accepts or stores an image; there is no
    # raw face data in this system to leak, retain past its purpose, or
    # count as a child's "image" under COPPA's training-data prohibition.
    embedding: List[float]
    model_version: str


class FaceEnrollmentResponse(BaseModel):
    student_id: UUID
    model_version: str
    enrolled_at: datetime


class FaceEnrollmentStatusResponse(BaseModel):
    student_id: UUID
    enrolled: bool
    model_version: Optional[str] = None
    enrolled_at: Optional[datetime] = None


async def _write_audit_log(session: AsyncSession, actor: User, action: str, entity_id: UUID, summary: str) -> None:
    session.add(
        AuditLog(
            id=uuid4(),
            actor_id=actor.id,
            actor_name=actor.full_name,
            action=action,
            entity_type="student",
            entity_id=entity_id,
            summary=summary,
        )
    )


async def _authorize_school_admin(school_id: UUID, current_user: User) -> None:
    if current_user.role == Role.super_admin:
        return
    if current_user.role != Role.school_admin or current_user.school_id != school_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this school")


@router.get("/schools", response_model=List[SchoolResponse])
async def list_schools(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    # No single-school scope to fall back to here, unlike `get_school` — a
    # school admin has exactly one school and reads it by id; this route only
    # makes sense for a role that oversees more than one.
    if current_user.role != Role.super_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only platform admins may list schools")
    result = await session.execute(select(School).order_by(School.name))
    return result.scalars().all()


@router.get("/schools/{school_id}", response_model=SchoolResponse)
async def get_school(
    school_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if current_user.role not in {Role.school_admin, Role.super_admin}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view school details")
    statement = select(School).where(School.id == school_id)
    result = await session.execute(statement)
    school = result.scalar_one_or_none()
    if not school:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="School not found")
    if current_user.role == Role.school_admin and current_user.school_id != school_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this school")
    return school


@router.post("/schools", response_model=SchoolResponse, status_code=status.HTTP_201_CREATED)
async def create_school(
    request: SchoolCreateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if current_user.role != Role.super_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only platform admins may create schools")

    school = School(
        id=uuid4(),
        name=request.name,
        code=request.code,
        region=request.region,
        district=request.district,
        address=request.address,
        phone=request.phone,
        email=request.email,
        headteacher=request.headteacher,
        levels=request.levels or [],
        status=SchoolStatus.active,
    )
    session.add(school)
    subscription = Subscription(
        id=uuid4(),
        school_id=school.id,
        plan=SubscriptionPlan.trial,
        status=SubscriptionStatus.trialing,
        started_at=datetime.utcnow(),
        current_period_end=datetime.utcnow() + timedelta(days=90),
        amount_minor=0,
        seats=0,
        auto_renew=False,
    )
    session.add(subscription)
    await session.commit()
    await session.refresh(school)
    return school


@router.patch("/schools/{school_id}/status")
async def update_school_status(
    school_id: UUID,
    request: SchoolStatusUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if current_user.role != Role.super_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only platform admins may update school status")
    statement = select(School).where(School.id == school_id)
    result = await session.execute(statement)
    school = result.scalar_one_or_none()
    if not school:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="School not found")
    school.status = request.status
    session.add(school)
    await session.commit()
    return {"status": "ok", "school_status": school.status}


@router.get("/schools/{school_id}/students", response_model=List[SchoolStudentResponse])
async def list_school_students(
    school_id: UUID,
    q: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None, alias="class"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    await _authorize_school_admin(school_id, current_user)
    query = select(Student).where(Student.school_id == school_id)
    if q:
        query = query.where(
            (Student.first_name.ilike(f"%{q}%"))
            | (Student.last_name.ilike(f"%{q}%"))
            | (Student.student_code.ilike(f"%{q}%"))
        )
    if class_name:
        query = query.where(Student.class_name == class_name)
    result = await session.execute(query)
    students = result.scalars().all()

    guardian_ids_by_student: dict[UUID, List[UUID]] = {}
    if students:
        student_ids = [s.id for s in students]
        guard_stmt = select(Guardianship).where(Guardianship.student_id.in_(student_ids))
        guard_result = await session.execute(guard_stmt)
        for guardianship in guard_result.scalars().all():
            guardian_ids_by_student.setdefault(guardianship.student_id, []).append(guardianship.parent_id)

    return [
        SchoolStudentResponse(
            id=s.id,
            student_code=s.student_code,
            first_name=s.first_name,
            last_name=s.last_name,
            class_name=s.class_name,
            level=s.level,
            allergies=s.allergies,
            dietary_notes=s.dietary_notes,
            guardian_ids=guardian_ids_by_student.get(s.id, []),
        )
        for s in students
    ]


@router.post("/schools/{school_id}/students", response_model=SchoolStudentResponse)
async def create_school_student(
    school_id: UUID,
    request: StudentCreateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    await _authorize_school_admin(school_id, current_user)
    existing_student = await session.execute(select(Student).where(Student.student_code == request.student_code))
    if existing_student.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Student code already exists")
    # A pupil is a students row and a wallets row, never a login — the kiosk
    # pivot (SPEC_KIOSK_AND_VERIFICATION.md §2) means nobody signs in as a
    # pupil any more, so minting a `users` row here was creating a live,
    # unused authentication surface for every single enrolment.
    student = Student(
        id=uuid4(),
        user_id=None,
        school_id=school_id,
        student_code=request.student_code,
        first_name=request.first_name,
        last_name=request.last_name,
        class_name=request.class_name,
        level=request.level,
        allergies=request.allergies or [],
        dietary_notes=request.dietary_notes,
        created_at=datetime.utcnow(),
    )
    session.add(student)
    await session.flush()  # `wallet.student_id` FKs to this row — must exist before the wallet insert
    # A student is unusable without one: no wallet means no top-up and no
    # order can ever be placed for them. Every other creation path (seed.py)
    # already pairs a student with a wallet; this is the real enrollment path
    # and was silently missing it.
    session.add(Wallet(id=uuid4(), student_id=student.id))
    await session.commit()
    await session.refresh(student)
    return student


@router.post("/schools/{school_id}/students/import", response_model=SchoolEnrollmentResult)
async def import_school_students(
    school_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    await _authorize_school_admin(school_id, current_user)
    decoded = (await file.read()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    imported = 0
    skipped = 0
    errors: List[str] = []
    for index, raw_row in enumerate(reader, start=1):
        # The app's own "Use sample data" template ships headers like "First
        # Name, Last Name, Class, Student ID" — human-readable, no `level`
        # column at all (the app derives level from the class name). Matching
        # only exact snake_case headers silently skipped every row of that
        # format, so normalise headers and accept the common aliases.
        row = {(k or "").strip().lower().replace(" ", "_"): v for k, v in raw_row.items()}
        student_code = row.get("student_code") or row.get("student_id")
        first_name = row.get("first_name")
        last_name = row.get("last_name")
        class_name = row.get("class_name") or row.get("class")
        level = row.get("level") or ("jhs" if class_name and "jhs" in class_name.lower() else "primary")
        if not all([student_code, first_name, last_name, class_name]):
            skipped += 1
            errors.append(f"row {index}: missing required fields")
            continue
        existing_student = await session.execute(select(Student).where(Student.student_code == student_code))
        if existing_student.scalar_one_or_none():
            skipped += 1
            errors.append(f"row {index}: student_code {student_code} already exists")
            continue
        student = Student(
            id=uuid4(),
            user_id=None,
            school_id=school_id,
            student_code=student_code,
            first_name=first_name,
            last_name=last_name,
            class_name=class_name,
            level=level,
            allergies=[item.strip() for item in (row.get("allergies") or "").split(",") if item.strip()],
            dietary_notes=row.get("dietary_notes") or None,
            created_at=datetime.utcnow(),
        )
        session.add(student)
        await session.flush()  # `wallet.student_id` FKs to this row — must exist before the wallet insert
        session.add(Wallet(id=uuid4(), student_id=student.id))
        imported += 1
    await session.commit()
    return SchoolEnrollmentResult(imported=imported, skipped=skipped, errors=errors)


@router.patch("/students/{student_id}", response_model=SchoolStudentResponse)
async def update_student(
    student_id: UUID,
    request: StudentUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    statement = select(Student).where(Student.id == student_id)
    result = await session.execute(statement)
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    school_id = student.school_id
    await _authorize_school_admin(school_id, current_user)
    update_data = request.dict(exclude_unset=True)
    for name, value in update_data.items():
        setattr(student, name, value)
    if request.first_name or request.last_name:
        user_stmt = select(User).where(User.id == student.user_id)
        user_result = await session.execute(user_stmt)
        user = user_result.scalar_one_or_none()
        if user:
            user.full_name = f"{request.first_name or student.first_name} {request.last_name or student.last_name}"
            session.add(user)
    session.add(student)
    await session.commit()
    await session.refresh(student)
    return student


async def _get_student_or_404(student_id: UUID, session: AsyncSession) -> Student:
    statement = select(Student).where(Student.id == student_id)
    result = await session.execute(statement)
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return student


@router.put("/students/{student_id}/face-enrollment", response_model=FaceEnrollmentResponse)
async def enroll_student_face(
    student_id: UUID,
    request: FaceEnrollmentRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Stores the one face embedding a kiosk compares a live capture against
    for this pupil. `PUT`, not `POST`: enrollment is idempotent by student —
    a re-capture (bad lighting, a model upgrade) replaces whatever was there
    rather than accumulating templates a kiosk could still be handed.

    The embedding must already be computed by the caller (a trusted
    school-admin device) before this request — nothing here accepts a photo,
    processes one, or has anywhere to store one.
    """
    student = await _get_student_or_404(student_id, session)
    await _authorize_school_admin(student.school_id, current_user)

    expected_dim = SUPPORTED_FACE_MODELS.get(request.model_version)
    if expected_dim is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model_version. Expected one of: {', '.join(SUPPORTED_FACE_MODELS)}",
        )
    if len(request.embedding) != expected_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{request.model_version} embeddings must have {expected_dim} values, got {len(request.embedding)}",
        )

    existing_stmt = select(FaceTemplate).where(FaceTemplate.student_id == student_id)
    existing_result = await session.execute(existing_stmt)
    template = existing_result.scalar_one_or_none()
    is_replacement = template is not None

    if template:
        template.embedding = request.embedding
        template.model_version = request.model_version
        template.enrolled_by = current_user.id
        template.enrolled_at = datetime.utcnow()
    else:
        template = FaceTemplate(
            id=uuid4(),
            student_id=student_id,
            embedding=request.embedding,
            model_version=request.model_version,
            enrolled_by=current_user.id,
            enrolled_at=datetime.utcnow(),
        )
    session.add(template)
    await _write_audit_log(
        session,
        current_user,
        "student.face_enrollment.replaced" if is_replacement else "student.face_enrollment.created",
        student_id,
        f"Face enrollment {'replaced' if is_replacement else 'created'} for {student.first_name} {student.last_name} ({request.model_version})",
    )
    await session.commit()
    await session.refresh(template)
    return FaceEnrollmentResponse(student_id=student_id, model_version=template.model_version, enrolled_at=template.enrolled_at)


@router.get("/students/{student_id}/face-enrollment", response_model=FaceEnrollmentStatusResponse)
async def get_student_face_enrollment_status(
    student_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Whether a pupil has an enrolled face template, for a school admin's
    own roster view. Deliberately never returns the embedding itself —
    nothing outside `POST /kiosks/verify`, which hands it to exactly the one
    kiosk mid-transaction for exactly the one matched student, should ever
    see it.
    """
    student = await _get_student_or_404(student_id, session)
    await _authorize_school_admin(student.school_id, current_user)

    template_stmt = select(FaceTemplate).where(FaceTemplate.student_id == student_id)
    template_result = await session.execute(template_stmt)
    template = template_result.scalar_one_or_none()
    if not template:
        return FaceEnrollmentStatusResponse(student_id=student_id, enrolled=False)
    return FaceEnrollmentStatusResponse(
        student_id=student_id, enrolled=True, model_version=template.model_version, enrolled_at=template.enrolled_at
    )


@router.delete("/students/{student_id}/face-enrollment", status_code=status.HTTP_204_NO_CONTENT)
async def delete_student_face_enrollment(
    student_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Removes a pupil's enrolled face template — the lever behind a parent
    asking for their child's biometric data to be deleted, which COPPA and
    Ghana's own Data Protection Act both treat as a right, not a courtesy.
    Kiosk face-matching for this pupil silently stops (the verify response
    just omits the embedding field); the code/PIN path is unaffected, since
    it was never able to depend on face enrollment existing in the first
    place.
    """
    student = await _get_student_or_404(student_id, session)
    await _authorize_school_admin(student.school_id, current_user)

    template_stmt = select(FaceTemplate).where(FaceTemplate.student_id == student_id)
    template_result = await session.execute(template_stmt)
    template = template_result.scalar_one_or_none()
    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No face enrollment on file")

    await session.delete(template)
    await _write_audit_log(
        session,
        current_user,
        "student.face_enrollment.deleted",
        student_id,
        f"Face enrollment deleted for {student.first_name} {student.last_name}",
    )
    await session.commit()
