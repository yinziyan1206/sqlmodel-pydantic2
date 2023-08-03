"""Private logic for creating models."""
from __future__ import annotations as _annotations

from typing import Any, Callable, ClassVar, Optional, Sequence, Union, cast
from warnings import warn

from pydantic import BaseModel, ConfigDict, PydanticUserError
from pydantic._internal._generics import PydanticGenericMetadata
from pydantic._internal._model_construction import ModelMetaclass
from pydantic._internal._repr import Representation
from pydantic.fields import AliasChoices, AliasPath
from pydantic.fields import FieldInfo as PydanticFieldInfo
from pydantic_core import PydanticUndefined
from sqlalchemy import Column, MetaData, inspect
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeMeta,
    Mapped,
    ORMDescriptor,
    RelationshipProperty,
    declared_attr,
    registry,
    relationship,
)
from sqlalchemy.orm.attributes import set_attribute
from sqlalchemy.orm.instrumentation import is_instrumented
from typing_extensions import dataclass_transform


class FieldInfo(PydanticFieldInfo):
    def __init__(self, default: Any = PydanticUndefined, **kwargs: Any) -> None:
        sa_column = kwargs.pop("sa_column", PydanticUndefined)
        super().__init__(default=default, **kwargs)
        self.sa_column = sa_column


def Field(  # noqa: C901
    default: Any = PydanticUndefined,
    *,
    default_factory: Optional[Callable[[], Any]] = PydanticUndefined,
    alias: Optional[str] = PydanticUndefined,
    alias_priority: Optional[int] = PydanticUndefined,
    validation_alias: Union[str, AliasPath, AliasChoices, None] = PydanticUndefined,
    serialization_alias: Optional[str] = PydanticUndefined,
    title: Optional[str] = PydanticUndefined,
    description: Optional[str] = PydanticUndefined,
    examples: Optional[list[Any]] = PydanticUndefined,
    exclude: Optional[bool] = PydanticUndefined,
    include: Optional[bool]= PydanticUndefined,
    discriminator: Optional[str] = PydanticUndefined,
    json_schema_extra: Union[dict[str, Any], Callable[[dict[str, Any]], None], None] = PydanticUndefined,
    frozen: Optional[bool] = PydanticUndefined,
    validate_default: Optional[bool] = PydanticUndefined,
    repr: Optional[bool] = PydanticUndefined,
    init_var: Optional[bool] = PydanticUndefined,
    kw_only: Optional[bool] = PydanticUndefined,
    pattern: Optional[str] = PydanticUndefined,
    strict: Optional[bool] = PydanticUndefined,
    gt: Optional[float] = PydanticUndefined,
    ge: Optional[float] = PydanticUndefined,
    lt: Optional[float] = PydanticUndefined,
    le: Optional[float] = PydanticUndefined,
    multiple_of: Optional[float] = PydanticUndefined,
    allow_inf_nan: Optional[bool] = PydanticUndefined,
    max_digits: Optional[int] = PydanticUndefined,
    decimal_places: Optional[int] = PydanticUndefined,
    min_length: Optional[int] = PydanticUndefined,
    max_length: Optional[int] = PydanticUndefined,
    sa_column: Optional[Column] = PydanticUndefined,
    **extra: Any  # type: ignore
) -> Any:
    const = extra.pop('const', None)  # type: ignore
    if const is not None:
        raise PydanticUserError('`const` is removed, use `Literal` instead', code='removed-kwargs')

    min_items = extra.pop('min_items', None)  # type: ignore
    if min_items is not None:
        warn('`min_items` is deprecated and will be removed, use `min_length` instead', DeprecationWarning)
        if min_length in (None, PydanticUndefined):
            min_length = min_items  # type: ignore

    max_items = extra.pop('max_items', None)  # type: ignore
    if max_items is not None:
        warn('`max_items` is deprecated and will be removed, use `max_length` instead', DeprecationWarning)
        if max_length in (None, PydanticUndefined):
            max_length = max_items  # type: ignore

    unique_items = extra.pop('unique_items', None)  # type: ignore
    if unique_items is not None:
        raise PydanticUserError(
            (
                '`unique_items` is removed, use `Set` instead'
                '(this feature is discussed in https://github.com/pydantic/pydantic-core/issues/296)'
            ),
            code='removed-kwargs',
        )

    allow_mutation = extra.pop('allow_mutation', None)  # type: ignore
    if allow_mutation is not None:
        warn('`allow_mutation` is deprecated and will be removed. use `frozen` instead', DeprecationWarning)
        if allow_mutation is False:
            frozen = True

    regex = extra.pop('regex', None)  # type: ignore
    if regex is not None:
        raise PydanticUserError('`regex` is removed. use `pattern` instead', code='removed-kwargs')

    if extra:
        warn(
            'Extra keyword arguments on `Field` is deprecated and will be removed. use `json_schema_extra` instead',
            DeprecationWarning,
        )
        if not json_schema_extra or json_schema_extra is PydanticUndefined:
            json_schema_extra = extra  # type: ignore

    if (
            validation_alias
            and validation_alias is not PydanticUndefined
            and not isinstance(validation_alias, (str, AliasChoices, AliasPath))
    ):
        raise TypeError('Invalid `validation_alias` type. it should be `str`, `AliasChoices`, or `AliasPath`')

    if serialization_alias in (PydanticUndefined, None) and isinstance(alias, str):
        serialization_alias = alias

    if validation_alias in (PydanticUndefined, None):
        validation_alias = alias
    return FieldInfo.from_field(
        default,
        default_factory=default_factory,
        alias=alias,
        alias_priority=alias_priority,
        validation_alias=validation_alias,
        serialization_alias=serialization_alias,
        title=title,
        description=description,
        examples=examples,
        exclude=exclude,
        include=include,
        discriminator=discriminator,
        json_schema_extra=json_schema_extra,
        frozen=frozen,
        validate_default=validate_default,
        repr=repr,
        init_var=init_var,
        kw_only=kw_only,
        pattern=pattern,
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
        min_length=min_length,
        max_length=max_length,
        sa_column=sa_column,
        **extra,
    )


class RelationshipInfo(Representation):
    def __init__(
        self,
        *,
        back_populates: Optional[str] = None,
        link_model: Optional[Any] = None,
        sa_relationship: Optional[RelationshipProperty] = None,  # type: ignore
    ) -> None:
        self.back_populates = back_populates
        self.link_model = link_model
        self.sa_relationship = sa_relationship


def Relationship(
    *,
    back_populates: Optional[str] = None,
    link_model: Optional[Any] = None,
    sa_relationship: Optional[RelationshipProperty] = None
) -> Any:
    if sa_relationship is None:
        rel_kwargs = {}
        if back_populates:
            rel_kwargs["back_populates"] = back_populates
        if link_model:
            ins = inspect(link_model)
            local_table = getattr(ins, "local_table")
            if local_table is None:
                raise RuntimeError(
                    "Couldn't find the secondary table for "
                    f"model {link_model}"
                )
            rel_kwargs["secondary"] = local_table
        sa_relationship = relationship(**rel_kwargs)

    relationship_info = RelationshipInfo(
        back_populates=back_populates,
        link_model=link_model,
        sa_relationship=sa_relationship
    )
    return relationship_info


default_registry = registry()


def get_column_from_field(field: FieldInfo) -> Union[Column, ORMDescriptor]:
    sa_column = getattr(field, "sa_column", PydanticUndefined)
    if isinstance(sa_column, Column):
        return sa_column.copy()
    return hybrid_property(lambda _: field.get_default())   # type: ignore


@dataclass_transform(kw_only_default=True, field_specifiers=(Field, FieldInfo))
class SQLModelMetaclass(ModelMetaclass, DeclarativeMeta):
    __sqlmodel_relationships__: dict[str, RelationshipInfo]

    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        __pydantic_generic_metadata__: PydanticGenericMetadata | None = None,
        __pydantic_reset_parent_namespace__: bool = True,
        **kwargs: Any,
    ) -> type:
        relationships: dict[str, RelationshipInfo] = {}
        dict_for_pydantic = {}

        original_annotations = dict(namespace.get("__annotations__", {}))
        pydantic_annotations = {}
        relationship_annotations = {}

        for k, v in namespace.items():
            if isinstance(v, RelationshipInfo):
                relationships[k] = v
            else:
                dict_for_pydantic[k] = v

        for k, v in original_annotations.items():
            if hasattr(v, '__origin__') and getattr(v, '__origin__') is Mapped:
                v = getattr(v, '__args__')[0]
            if k in relationships:
                relationship_annotations[k] = v
            else:
                pydantic_annotations[k] = v

        dict_used = {
            **dict_for_pydantic,
            "__weakref__": None,
            "__sqlmodel_relationships__": relationships,
            "__annotations__": pydantic_annotations,
        }

        allowed_config_kwargs: set[str] = {
            key
            for key in dir(ConfigDict)
            if not (
                key.startswith("__") and key.endswith("__")
            )  # skip dunder methods and attributes
        }
        pydantic_kwargs = kwargs.copy()
        config_kwargs = {
            key: pydantic_kwargs.pop(key)
            for key in pydantic_kwargs.keys() & allowed_config_kwargs
        }

        new_cls = super().__new__(
            mcs,
            cls_name,
            bases,
            dict_used,
            __pydantic_generic_metadata__,
            __pydantic_reset_parent_namespace__,
            **config_kwargs
        )

        new_cls.__annotations__ = {
            **relationship_annotations,
            **pydantic_annotations,
            **new_cls.__annotations__,
        }

        config_registry = kwargs.get('registry', None)
        config_table = kwargs.get('table', None)

        if config_table:
            new_cls.model_config["table"] = True
            new_cls.model_config['from_attributes'] = True
            for k, v in new_cls.model_fields.items():
                setattr(new_cls, k, get_column_from_field(v))

        if config_registry:
            config_registry = cast(registry, config_registry)
            setattr(new_cls, "__abstract__", True)
            setattr(new_cls, "_sa_registry", config_registry)
            setattr(new_cls, "metadata", config_registry.metadata)

        return new_cls

    def __setattr__(self, name: str, value: Any) -> None:
        if self.model_config.get("table", False):
            DeclarativeMeta.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if self.model_config.get("table", False):
            DeclarativeMeta.__delattr__(self, name)
        else:
            super().__delattr__(name)

    def __init__(
        cls, classname: str, bases: tuple[type, ...], dict_: dict[str, Any], **kw: Any
    ) -> None:
        base_is_table = False
        for base in bases:
            if base.model_config.get("table", False):
                base_is_table = True
                break
        if cls.model_config.get("table", False) and not base_is_table:
            dict_used = dict_.copy()
            for field_name, field_value in cls.model_fields.items():
                dict_used[field_name] = get_column_from_field(field_value)
            for rel_name, rel_info in cls.__sqlmodel_relationships__.items():
                if not rel_info.sa_relationship:
                    raise RuntimeError(
                        f"Couldn't find the relationship for {rel_name}"
                    )
                dict_used[rel_name] = rel_info.sa_relationship
            DeclarativeMeta.__init__(cls, classname, bases, dict_used, **kw)


class SQLModel(BaseModel, metaclass=SQLModelMetaclass, registry=default_registry):
    __slots__ = '__weakref__'
    __tablename__: ClassVar[Union[str, Callable[..., str]]]
    __sqlmodel_relationships__: ClassVar[dict[str, RelationshipProperty]]
    __name__: ClassVar[str]
    metadata: ClassVar[MetaData]

    model_config = ConfigDict(from_attributes=True)

    def __new__(cls, *args, **kwargs):
        new_object = super().__new__(cls)
        # SQLAlchemy doesn't call __init__ on the base class
        # Ref: https://docs.sqlalchemy.org/en/14/orm/constructors.html
        # Set __fields_set__ here, that would have been set when calling __init__
        # in the Pydantic model so that when SQLAlchemy sets attributes that are
        # added (e.g. when querying from DB) to the __fields_set__, this already exists
        object.__setattr__(new_object, "__pydantic_fields_set__", set())
        return new_object

    def __init__(__pydantic_self__, **data: Any) -> None:  # type: ignore
        _sa_instance = __pydantic_self__.__dict__.get('_sa_instance_state', None)
        super().__init__(**data)
        setattr(__pydantic_self__, '_sa_instance_state', _sa_instance)

        for key, value in __pydantic_self__.__dict__.items():
            setattr(__pydantic_self__, key, value)

        non_pydantic_keys = data.keys() - __pydantic_self__.__dict__.keys()
        for key in non_pydantic_keys:
            if key in __pydantic_self__.__sqlmodel_relationships__:
                setattr(__pydantic_self__, key, data[key])

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_sa_instance_state":
            self.__dict__[name] = value
        else:
            # Set in SQLAlchemy, before Pydantic to trigger events and updates
            if self.model_config.get("table", False) and is_instrumented(self, name):
                set_attribute(self, name, value)
            if name not in self.__sqlmodel_relationships__:
                super().__setattr__(name, value)

    def __getattr__(self, item) -> Any:
        if item == '_sa_instance_state' and '_sa_instance_state' in self.__dict__:
            return self.__dict__['_sa_instance_state']
        return super().__getattr__(item)

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    def __repr_args__(self) -> Sequence[tuple[Optional[str], Any]]:
        # Don't show SQLAlchemy private attributes
        return [(k, v) for k, v in self.__dict__.items() if not k.startswith("_sa_")]

    @property
    def __fields_set__(self):
        return self.__pydantic_fields_set__
