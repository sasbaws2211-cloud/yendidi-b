from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import Column, DateTime, JSON, String, UniqueConstraint, text
from sqlmodel import Field, SQLModel


class Role(str, Enum):
    student = "student"
    parent = "parent"
    vendor = "vendor"
    school_admin = "school_admin"
    super_admin = "super_admin"


class User(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    role: Role
    full_name: str
    email: Optional[str] = Field(index=True, unique=True)
    phone: Optional[str] = Field(index=True, unique=True)
    password_hash: Optional[str]
    school_id: Optional[UUID] = Field(foreign_key="school.id")
    is_active: bool = Field(default=True)
    mfa_secret: Optional[str]
    last_login_at: Optional[datetime]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    notification_preferences: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, server_default=text("'{}'")),
    )


class DeviceRegistration(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id")
    device_token: str = Field(index=True)
    platform: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RefreshToken(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id")
    token_hash: str
    device_id: str
    expires_at: datetime = Field(sa_type=DateTime(timezone=True))
    revoked_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
    replaced_by: Optional[UUID] = Field(foreign_key="refreshtoken.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))


class OTPPurpose(str, Enum):
    login = "login"
    password_reset = "password_reset"
    parent_onboarding = "parent_onboarding"


class OneTimePassword(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id")
    recipient: str
    purpose: OTPPurpose
    code_hash: str
    expires_at: datetime = Field(sa_type=DateTime(timezone=True))
    used_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
    attempts: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))


class SchoolStatus(str, Enum):
    active = "active"
    suspended = "suspended"
    pending = "pending"


class School(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    name: str
    code: str = Field(sa_column_kwargs={"unique": True})
    region: str
    district: str
    address: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    headteacher: Optional[str]
    levels: List[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False, server_default=text("'[]'")))
    status: SchoolStatus = Field(default=SchoolStatus.active)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SubscriptionPlan(str, Enum):
    trial = "trial"
    monthly = "monthly"
    annual = "annual"


class SubscriptionStatus(str, Enum):
    trialing = "trialing"
    active = "active"
    past_due = "past_due"
    cancelled = "cancelled"
    expired = "expired"


class Subscription(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    school_id: UUID = Field(foreign_key="school.id", unique=True)
    plan: SubscriptionPlan
    status: SubscriptionStatus
    started_at: datetime
    current_period_end: datetime
    amount_minor: int = Field(default=0)
    seats: int = Field(default=0)
    auto_renew: bool = Field(default=True)


class InvoiceStatus(str, Enum):
    paid = "paid"
    pending = "pending"
    failed = "failed"


class Invoice(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    subscription_id: UUID = Field(foreign_key="subscription.id")
    amount_minor: int
    status: InvoiceStatus
    period_start: datetime
    period_end: datetime
    method: Optional[str]
    reference: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Student(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    # Nullable since the kiosk pivot (SPEC_KIOSK_AND_VERIFICATION.md): a pupil
    # is a students row and a wallets row, never a login, so nothing mints a
    # `user` for one any more. Old rows keep whatever `user_id` they already
    # had; new ones are enrolled with none.
    user_id: Optional[UUID] = Field(default=None, foreign_key="user.id", unique=True)
    school_id: UUID = Field(foreign_key="school.id")
    student_code: str
    first_name: str
    last_name: str
    class_name: str
    level: str
    allergies: List[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False, server_default=text("'[]'")))
    dietary_notes: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FaceTemplate(SQLModel, table=True):
    """The one enrolled face embedding for a pupil, used to confirm identity
    at a kiosk — a 1:1 check against the single candidate `student_code`
    already names, never a search across the school's faces.

    Holds only the embedding vector a trusted device computed at enrollment
    time, never the source photo: a raw face image is not persisted anywhere
    in this system. That single choice is what keeps a breach of this table
    low-stakes (an embedding is not a viewable face) and what keeps
    enrollment clear of COPPA's prohibition on retaining children's images
    for anything resembling model training — there is no image to retain.

    `student_id` is unique: re-enrollment (a bad capture, a model upgrade)
    replaces the row rather than accumulating old templates a kiosk could
    still be handed.
    """

    id: Optional[UUID] = Field(default=None, primary_key=True)
    student_id: UUID = Field(foreign_key="student.id", unique=True)
    embedding: List[float] = Field(sa_column=Column(JSON, nullable=False))
    # Names the model the embedding was computed with (e.g. "arcface-r100-v1")
    # rather than assuming one forever — two embeddings from different models
    # are not comparable, so a kiosk has to know before it trusts a match.
    model_version: str
    enrolled_by: UUID = Field(foreign_key="user.id")
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)


class Parent(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", unique=True)
    full_name: str
    phone: str
    alt_phone: Optional[str]
    occupation: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Guardianship(SQLModel, table=True):
    parent_id: UUID = Field(foreign_key="parent.id", primary_key=True)
    student_id: UUID = Field(foreign_key="student.id", primary_key=True)
    is_primary: bool = Field(default=False)
    linked_at: datetime = Field(default_factory=datetime.utcnow)


class VendorStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    suspended = "suspended"
    rejected = "rejected"


class Vendor(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", unique=True)
    business_name: str
    owner_name: str
    phone: str
    email: Optional[str]
    description: Optional[str]
    status: VendorStatus = Field(default=VendorStatus.pending)
    rating: float = Field(default=0)
    rating_count: int = Field(default=0)
    opens_at_minutes: int
    closes_at_minutes: int
    accepting_orders: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VendorSchool(SQLModel, table=True):
    vendor_id: UUID = Field(foreign_key="vendor.id", primary_key=True)
    school_id: UUID = Field(foreign_key="school.id", primary_key=True)
    approved_at: Optional[datetime]


class VendorCertification(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    vendor_id: UUID = Field(foreign_key="vendor.id")
    name: str
    authority: str
    number: str
    document_key: Optional[str]
    issued_at: datetime
    expires_at: datetime
    verified: bool = Field(default=False)
    verified_by: Optional[UUID]
    verified_at: Optional[datetime]


class PayoutStatus(str, Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"


class VendorPayout(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    vendor_id: UUID = Field(foreign_key="vendor.id")
    school_id: UUID = Field(foreign_key="school.id")
    amount_minor: int
    status: PayoutStatus = Field(default=PayoutStatus.pending)
    reference: Optional[str]
    processed_at: Optional[datetime]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FoodCategory(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    snacks = "snacks"
    drinks = "drinks"
    desserts = "desserts"
    fruits = "fruits"
    healthy = "healthy"
    local = "local"


class MenuItem(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    vendor_id: UUID = Field(foreign_key="vendor.id")
    name: str
    description: Optional[str]
    category: FoodCategory
    tags: List[FoodCategory] = Field(default_factory=list, sa_column=Column(JSON, nullable=False, server_default=text("'[]'")))
    price_minor: int
    art_key: str
    ingredients: List[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False, server_default=text("'[]'")))
    available: bool = Field(default=True)
    stock_count: int = Field(default=0)
    low_stock_threshold: int = Field(default=8)
    prep_minutes: int = Field(default=8)
    kcal: Optional[int]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Wallet(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    student_id: UUID = Field(foreign_key="student.id", unique=True)
    balance_minor: int = Field(default=0)
    frozen: bool = Field(default=False)
    daily_limit_minor: Optional[int]
    weekly_limit_minor: Optional[int]
    blocked_categories: List[FoodCategory] = Field(default_factory=list, sa_column=Column(JSON, nullable=False, server_default=text("'[]'")))
    low_balance_threshold_minor: int = Field(default=2000)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TransactionType(str, Enum):
    topup = "topup"
    purchase = "purchase"
    refund = "refund"
    adjustment = "adjustment"
    reversal = "reversal"


class WalletTransaction(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    wallet_id: UUID = Field(foreign_key="wallet.id")
    student_id: UUID = Field(foreign_key="student.id")
    type: TransactionType
    amount_minor: int
    balance_after_minor: int
    description: str
    method: Optional[str]
    reference: str
    order_id: Optional[UUID]
    actor_id: Optional[UUID]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TopupStatus(str, Enum):
    pending = "pending"
    succeeded = "succeeded"
    failed = "failed"
    expired = "expired"


class Topup(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("wallet_id", "initiated_by", "idempotency_key", name="uq_topup_idempotency"),)

    id: Optional[UUID] = Field(default=None, primary_key=True)
    wallet_id: UUID = Field(foreign_key="wallet.id")
    initiated_by: UUID = Field(foreign_key="user.id")
    amount_minor: int
    method: str
    processor: str
    processor_ref: Optional[str]
    status: TopupStatus = Field(default=TopupStatus.pending)
    idempotency_key: str
    authorization_url: Optional[str] = None
    processor_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    failure_reason: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    settled_at: Optional[datetime]


class OrderStatus(str, Enum):
    # The wallet is debited at checkout, so "paid" is the whole order until
    # the vendor acts on it — no kitchen-stage pipeline in between. From
    # there the vendor has exactly one decision to make: confirm it, or
    # cancel it (which refunds the wallet in full). There is no
    # student-initiated cancel and no separate "rejected" state — one vendor
    # action, one terminal outcome either way.
    paid = "paid"
    confirmed = "confirmed"
    cancelled = "cancelled"


class Order(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    code: str
    student_id: UUID = Field(foreign_key="student.id")
    school_id: UUID = Field(foreign_key="school.id")
    vendor_id: UUID = Field(foreign_key="vendor.id")
    subtotal_minor: int
    service_fee_minor: int
    total_minor: int
    status: OrderStatus = Field(default=OrderStatus.paid)
    pickup_slot: str
    note: Optional[str]
    idempotency_key: Optional[str] = Field(default=None, sa_column=Column(String, unique=True, nullable=True))
    placed_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class OrderLine(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    order_id: UUID = Field(foreign_key="order.id")
    menu_item_id: Optional[UUID] = Field(foreign_key="menuitem.id")
    name: str
    art_key: str
    unit_price_minor: int
    quantity: int


class OrderEvent(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    order_id: UUID = Field(foreign_key="order.id")
    status: OrderStatus
    actor_id: Optional[UUID]
    note: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Notification(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id")
    kind: str
    title: str
    body: str

    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True)
    )

    read_at: Optional[datetime]

    channels: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(
            JSON,
            nullable=False,
            server_default=text("'{}'")
        )
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)


class Announcement(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    school_id: UUID = Field(foreign_key="school.id")
    author_id: UUID = Field(foreign_key="user.id")
    title: str
    body: str
    audience: List[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False, server_default=text("'[]'")))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WebhookEvent(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    provider: str
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False, server_default=text("'{}'")))
    received_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLog(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    actor_id: Optional[UUID]
    actor_name: str
    action: str
    entity_type: str
    entity_id: Optional[UUID]
    summary: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FavoriteTargetType(str, Enum):
    menu_item = "menu_item"
    vendor = "vendor"


class Favorite(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("student_id", "target_type", "target_id", name="uq_favorite_target"),)

    id: Optional[UUID] = Field(default=None, primary_key=True)
    student_id: UUID = Field(foreign_key="student.id")
    target_type: FavoriteTargetType
    target_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Invitation(SQLModel, table=True):
    """Credential-issuance token for vendors and school admins — see
    docs/SPEC_ACCOUNT_PROVISIONING.md §3.4. Parents need no invitation; they
    sign in by OTP the moment their account is created.

    The token itself is never stored — only its SHA-256 hash — so a leaked
    database row is not a working account-takeover link, the same treatment
    already given to `RefreshToken.token_hash`.
    """

    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id")
    token_hash: str = Field(index=True, unique=True)
    expires_at: datetime
    consumed_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VendorSubmissionStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class VendorSubmission(SQLModel, table=True):
    """A school's proposal that a vendor be allowed to sell to its pupils.

    Deliberately not a vendor — only a platform admin can create one of those
    (docs/SPEC_ACCOUNT_PROVISIONING.md §2.3). Approval creates the vendor in
    `pending`, still needing food-safety documents.
    """

    id: Optional[UUID] = Field(default=None, primary_key=True)
    school_id: UUID = Field(foreign_key="school.id")
    submitted_by_user_id: UUID = Field(foreign_key="user.id")
    submitted_by_name: str
    business_name: str
    owner_name: str
    phone: str
    email: Optional[str] = None
    description: Optional[str] = None
    opens_at_minutes: int
    closes_at_minutes: int
    status: VendorSubmissionStatus = Field(default=VendorSubmissionStatus.pending)
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_by_name: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_note: Optional[str] = None
    vendor_id: Optional[UUID] = Field(default=None, foreign_key="vendor.id")


class GuardianLinkStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class GuardianLinkRequest(SQLModel, table=True):
    """A parent's claim to a pupil, gated on the school's confirmation.

    Replaces linking on a bare student-code match (BACKEND_HANDOVER.md §2.1a):
    a code is a school prefix and four digits, printed on a card a child
    carries all day, so an instant link let any account walk the range and
    become guardian to a stranger's child. The school holds the register and
    is the only party that can actually verify which adult belongs to which
    pupil — not the platform admin, who would only be rubber-stamping.
    """

    id: Optional[UUID] = Field(default=None, primary_key=True)
    parent_id: UUID = Field(foreign_key="parent.id")
    student_id: UUID = Field(foreign_key="student.id")
    school_id: UUID = Field(foreign_key="school.id")
    status: GuardianLinkStatus = Field(default=GuardianLinkStatus.pending)
    reviewed_by_name: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reject_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GuardianLinkAttempt(SQLModel, table=True):
    """One row per `POST /guardian-link-requests` call, valid code or not.

    A request for an unknown code creates no `GuardianLinkRequest` — there is
    nothing to review — so without a separate record of the attempt itself,
    rate-limiting has nothing to count and an attacker can walk the ~1,000
    codes per school for free. This table exists only to be counted against
    over a rolling window.
    """

    id: Optional[UUID] = Field(default=None, primary_key=True)
    parent_id: UUID = Field(foreign_key="parent.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KioskStatus(str, Enum):
    pending = "pending"
    active = "active"
    revoked = "revoked"


class Kiosk(SQLModel, table=True):
    """A shared ordering device belonging to a school, not a person
    (SPEC_KIOSK_AND_VERIFICATION.md §2.1). Provisioned with a one-time pairing
    code; the device exchanges it once for a long-lived token and the code is
    never usable again. A kiosk token can read only its own school's menu,
    call verify, and place orders that carry a spent verification token — it
    cannot read a pupil record, a wallet, or another school's anything, on
    the assumption that a device in a corridor is compromised.
    """

    id: Optional[UUID] = Field(default=None, primary_key=True)
    school_id: UUID = Field(foreign_key="school.id")
    label: str
    status: KioskStatus = Field(default=KioskStatus.pending)
    pairing_code_hash: Optional[str] = None
    pairing_code_expires_at: Optional[datetime] = None
    device_token_hash: Optional[str] = Field(default=None, unique=True)
    # Set once at provisioning and checked entirely device-side by
    # `POST /kiosks/exit`. Short and shared knowledge among on-site staff
    # rather than a personal credential — leaking it stops kiosk service (a
    # nuisance), never data, since the device holds none. It exists so a
    # device can be taken out of service by whoever is standing next to it,
    # without needing an admin's JWT.
    exit_pin_hash: Optional[str] = None
    paired_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KioskVerification(SQLModel, table=True):
    """The single-use token minted by `POST /kiosks/verify` and spent by the
    order it authorises (§9.3–9.4). Short-lived and one-shot on purpose: it is
    a checkout context scoped to one transaction, not a standing credential —
    that distinction is what makes it safe to hand to a device in a corridor.
    """

    id: Optional[UUID] = Field(default=None, primary_key=True)
    token_hash: str = Field(index=True, unique=True)
    kiosk_id: UUID = Field(foreign_key="kiosk.id")
    student_id: UUID = Field(foreign_key="student.id")
    expires_at: datetime
    used_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IdempotencyRecord(SQLModel, table=True):
    """Generic idempotency store for account-provisioning endpoints, whose
    natural key (an admin creating an account) has no `wallet_id`-style scope
    to piggyback on the way `Topup` does. A repeat with the same
    (actor, scope, key) returns the first response verbatim instead of
    creating a second account — see docs/SPEC_ACCOUNT_PROVISIONING.md §2.5.
    """

    __table_args__ = (UniqueConstraint("actor_id", "scope", "idempotency_key", name="uq_idempotency_scope_key"),)

    id: Optional[UUID] = Field(default=None, primary_key=True)
    actor_id: UUID = Field(foreign_key="user.id")
    scope: str
    idempotency_key: str
    response_body: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status_code: int = Field(default=201)
    created_at: datetime = Field(default_factory=datetime.utcnow)
