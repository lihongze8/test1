from response_format.base import (
    _ensure_class,
    _parse_model_to_json_output_prompt,
    _parse_model_to_response_format_prompt
)


@_ensure_class
def format_prompt_getter(cls):
    """
    为装饰的类添加结构化输出提示的方法。

    # Args:
        cls: 需要添加提示的类模型。

    # Returns:
        cls: 返回添加了输出提示的类模型。
    """
    @classmethod
    def get_response_format_prompt(cls):
        return _parse_model_to_response_format_prompt(cls)
    @classmethod
    def get_json_output_prompt(cls):
        return _parse_model_to_json_output_prompt(cls)
    setattr(cls, "get_response_format_prompt", get_response_format_prompt)
    setattr(cls, "get_json_output_prompt", get_json_output_prompt)
    return cls