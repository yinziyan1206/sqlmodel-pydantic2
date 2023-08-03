# sqlmodel-pydantic2

**SQLModel** for **Pydantic2** and **SQLAlchemy2**

# Usage
```python

class Hero(SQLModel, table=True):
    id: Mapped[int | None] = Field(default=None, title='ID', sa_column=Column(Integer, primary_key=True))
    name: Mapped[str] = Field(default='', title='hero's name', sa_column=Column(String(64), nullable=False))
    secret_name: str = Field(default='', title='hero's name')   # not a column
    age: Mapped[int] = Field(default=0, title='hero's age', sa_column=Column(Integer, nullable=False))

    @computed_field
    @property
    def display_name(self) -> str:
        return self.name + self.secret_name
```

# Link

* **SQLModel**: https://github.com/tiangolo/sqlmodel
* **Pydantic**: https://github.com/pydantic/pydantic
* **SQLAlchemy**: https://github.com/sqlalchemy/sqlalchemy
