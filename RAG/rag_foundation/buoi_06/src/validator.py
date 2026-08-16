import re
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator

class DocumentMetadata(BaseModel):
    """
    Schema đại diện cho siêu dữ liệu (metadata) của một văn bản.
    """
    so_hieu: str = Field(..., description="Số hiệu văn bản (vd: 15/2026/QĐ-KTNB)")
    tieu_de: str = Field(..., description="Tiêu đề văn bản")
    ngay_ban_hanh: Optional[str] = Field(None, description="Ngày ban hành theo định dạng YYYY-MM-DD")
    con_hieu_luc: bool = Field(True, description="Trạng thái hiệu lực của văn bản")
    nguon: str = Field("unknown", description="Nguồn dữ liệu (vd: file PDF, hệ thống nội bộ)")

    @field_validator("ngay_ban_hanh")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        
        # Hỗ trợ định dạng YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            try:
                datetime.strptime(v, "%Y-%m-%d")
                return v
            except ValueError:
                raise ValueError(f"Ngày ban hành không hợp lệ: {v}")
        
        # Hỗ trợ định dạng DD/MM/YYYY và chuyển đổi về YYYY-MM-DD
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", v):
            try:
                parsed_date = datetime.strptime(v, "%d/%m/%Y")
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Ngày ban hành không hợp lệ: {v}")

        raise ValueError("Định dạng ngày ban hành không được hỗ trợ. Sử dụng YYYY-MM-DD hoặc DD/MM/YYYY.")

    @model_validator(mode="after")
    def check_empty_fields(self):
        if not self.so_hieu.strip():
            raise ValueError("Số hiệu văn bản không được để trống")
        if not self.tieu_de.strip():
            raise ValueError("Tiêu đề văn bản không được để trống")
        return self


class DocumentChunk(BaseModel):
    """
    Schema đại diện cho một đoạn văn bản (chunk) sau khi cắt ra từ Document Loader,
    sẵn sàng để đưa vào quá trình Embedding.
    """
    content: str = Field(..., description="Nội dung văn bản của chunk")
    metadata: DocumentMetadata = Field(..., description="Siêu dữ liệu đi kèm của văn bản gốc")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Nội dung văn bản (content) của chunk không được để trống")
        # Cơ chế làm sạch các ký tự đặc biệt thừa thãi có thể thêm vào đây
        # Ví dụ xóa khoảng trắng thừa:
        v = re.sub(r"\s+", " ", v)
        return v
