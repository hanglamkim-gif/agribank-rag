"""Cấu hình các vai trò chuẩn trong hệ thống RBAC."""

ROLES = {
    "ADMIN": "Admin",
    "HR": "HR_Manager",
    "RISK": "Risk_Officer",
    "STAFF": "Staff",
    "GUEST": "Guest"
}

ALL_ROLES = list(ROLES.values())