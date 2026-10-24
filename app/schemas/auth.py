"""Auth schemas.

覆盖注册 / 登录 / 令牌的请求与响应模型.
"""
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """注册请求体."""

    username: str = Field(..., min_length=3, max_length=32, description="用户名 (长度 3-32 字符)")
    email: EmailStr = Field(..., description="邮箱; 须为合法格式")
    password: str = Field(..., min_length=6, max_length=128, description="密码 (长度 6-128 字符)")


class LoginRequest(BaseModel):
    """登录请求体."""

    username: str = Field(..., description="登录用户名")
    password: str = Field(..., description="登录密码")


class TokenResponse(BaseModel):
    """登录 / 注册 / dev-token 成功后下发的令牌.

    access_token 会同时以 HttpOnly cookie (pilkb_token) 和 Bearer 两种形式
    返回: 前端依赖 cookie 自动携带鉴权, 脚本与第三方工具则用 Bearer 头.
    """

    access_token: str = Field(..., description="JWT 形式的访问令牌")
    token_type: str = Field(default="bearer", description="令牌的类型")
    expires_in: int = Field(..., description="有效期 (秒), 等于 jwt_expire_minutes * 60")
