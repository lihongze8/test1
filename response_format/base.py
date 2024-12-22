import json
import logging
import re
import ast
import inspect
from pydantic import BaseModel
from json_repair import repair_json

log = logging.getLogger(__name__)


def _try_parse_ast_to_json(function_string: str) -> tuple[str, dict]:
    """
    尝试使用AST解析非标准JSON格式的函数调用字符串，并转换为标准JSON对象。

    # Args:
        function_string (str): 需要解析的函数调用字符串，例如 "func(arg1={'key': 'value'}, arg2='value')"

    # Returns:
        tuple[str, dict]: 返回一个元组，其中第一个元素是解析过程的详细信息，第二个元素是解析后的JSON对象。

    # Details:
        - 使用 `ast.parse` 解析输入字符串，假设其为函数调用形式。
        - 遍历AST节点，提取函数名、参数名和参数值。
        - 构建一个标准的JSON对象，并生成解析过程的详细信息。
        - 该方法作为备选解析方案，用于处理 `json.loads` 无法解析的非标准JSON字符串。
    """
    tree = ast.parse(str(function_string).strip())
    ast_info = ""
    json_result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = node.func.id
            args = {kw.arg: kw.value for kw in node.keywords}
            ast_info += f"Function Name: {function_name}\r\n"
            for arg, value in args.items():
                ast_info += f"Argument Name: {arg}\n"
                ast_info += f"Argument Value: {ast.dump(value)}\n"
                json_result[arg] = ast.literal_eval(value)

    return ast_info, json_result


def _try_parse_json_object(input: str) -> tuple[str, dict]:
    """
    尝试解析并修复非标准JSON字符串，返回标准JSON对象。

    # Args:
        input (str): 需要解析的JSON字符串，可能包含非标准格式或Markdown框架。

    # Returns:
        tuple[str, dict]: 返回一个元组，其中第一个元素是原始或修复后的JSON字符串，第二个元素是解析后的JSON对象。

    # Details:
        - 首先尝试使用 `json.loads` 直接解析输入字符串。
        - 如果解析失败，进行字符串清理和修复，去除多余字符和Markdown框架。
        - 再次尝试解析清理后的字符串，如果仍失败，使用 `repair_json` 进行修复。
        - 作为备选方案，调用 `_try_parse_ast_to_json` 处理非标准JSON格式。
        - 确保返回结果为字典类型，处理异常情况并记录日志。
    """
    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        log.warning("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```"):
        input = input[len("```"):]
    if input.startswith("```json"):
        input = input[len("```json"):]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        json_info = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:

            if len(json_info) < len(input):
                json_info, result = _try_parse_ast_to_json(input)
            else:
                result = json.loads(json_info)

        except json.JSONDecodeError:
            log.exception("error loading json, json=%s", input)
            return json_info, {}
        else:
            if not isinstance(result, dict):
                log.exception("not expected dict type. type=%s:", type(result))
                return json_info, {}
            return json_info, result
    else:
        return input, result


def _parse_model_schema(model: type[BaseModel]):
    """
    解析模型的JSON schema。

    # Args:
        model (type[BaseModel]): 需要解析的模型类。

    # Returns:
        dict: 返回模型的JSON schema。
    """
    schema = model.model_json_schema()
    return _parse_schema_properties(schema, schema)


def _parse_schema_properties(schema, root_schema):
    """
    解析schema属性。

    # Args:
        schema (dict): 当前schema。
        root_schema (dict): 根schema。

    # Returns:
        dict: 返回解析后的属性字典。
    """
    result = {}
    properties = schema.get("properties", {})

    for field_name, field_info in properties.items():
        if field_info.get("type") == "array":
            items = field_info.get("items", {})
            if "$ref" in items:
                ref_name = items["$ref"].split("/")[-1]
                nested_schema = root_schema.get("$defs", {}).get(
                    ref_name) or root_schema.get("definitions", {}).get(ref_name)
                if nested_schema:
                    nested_properties = _parse_schema_properties(
                        nested_schema, root_schema)
                    result[field_name] = [nested_properties]
                else:
                    result[field_name] = []
            else:
                nested_properties = _parse_schema_properties(
                    items, root_schema)
                result[field_name] = [nested_properties]
        elif field_info.get("type") == "object":
            nested_properties = _parse_schema_properties(
                field_info, root_schema)
            result[field_name] = nested_properties
        else:
            result[field_name] = field_info.get("description", "")

    return result


def _add_ellipsis_for_display(parsed_schema, schema, root_schema):
    """
    为显示添加省略号。

    # Args:
        parsed_schema (dict): 解析后的schema。
        schema (dict): 当前schema。
        root_schema (dict): 根schema。

    # Returns:
        dict: 返回添加省略号后的schema。
    """
    result = {}
    properties = schema.get("properties", {})

    for field_name, field_info in properties.items():
        if field_info.get("type") == "array":
            items = field_info.get("items", {})
            if "$ref" in items:
                ref_name = items["$ref"].split("/")[-1]
                nested_schema = root_schema.get("$defs", {}).get(
                    ref_name) or root_schema.get("definitions", {}).get(ref_name)
                if nested_schema:
                    nested_properties = _add_ellipsis_for_display(
                        parsed_schema[field_name][0], nested_schema, root_schema)
                    result[field_name] = [
                        nested_properties,
                        f"(... more {field_name}.{field_info.get('description', '')} ...)"
                    ]
                else:
                    result[field_name] = [field_info.get("description", "")]
            else:
                nested_properties = _add_ellipsis_for_display(
                    parsed_schema[field_name][0], items, root_schema)
                result[field_name] = [
                    nested_properties,
                    f"(... more {field_name}.{field_info.get('description', '')} ...)"
                ]
        elif field_info.get("type") == "object":
            nested_properties = _add_ellipsis_for_display(
                parsed_schema[field_name], field_info, root_schema)
            result[field_name] = nested_properties
        else:
            result[field_name] = parsed_schema[field_name]

    return result

def _parse_model_to_json_output_prompt(model: BaseModel) -> str:
    """
    生成模型的JSON输出提示。

    # Args:
        model (BaseModel): 需要生成提示的模型实例。

    # Returns:
        str: 返回JSON输出提示字符串。
    """
    parsed_test_schema = _parse_model_schema(model)
    schema = model.model_json_schema()
    display_schema = _add_ellipsis_for_display(
        parsed_test_schema, schema, schema)
    prompt = json.dumps(display_schema, ensure_ascii=False, indent=4)
    prompt = re.sub(
        r'"(\(\.{3} more [^"]+\.{3}\))"', r'\1', prompt)
    return prompt

PROMPT_TEMPLATE = """Kindly provide the response if you have the answer and output them in JSON format.

EXAMPLE JSON OUTPUT AND DESCRIPTIONS:
{model_json_output_prompt}
"""

def _parse_model_to_response_format_prompt(model: BaseModel) -> str:
    """
    生成模型的响应格式提示。

    # Args:
        model (BaseModel): 需要生成提示的模型实例。

    # Returns:
        str: 返回响应格式提示字符串。
    """
    global PROMPT_TEMPLATE
    prompt = _parse_model_to_json_output_prompt(model)
    return PROMPT_TEMPLATE.format(model_json_output_prompt=prompt)

def _ensure_class(decorator):
    """
    确保装饰器只能用于类型为BaseModel的类模型上的包装器。

    # Args:
        decorator: 需要被包装的装饰器函数。

    # Returns:
        包装后的装饰器，只对类型为BaseModel的类模型有效。
    """
    def wrapper(cls):
        if not inspect.isclass(cls) or not issubclass(cls, BaseModel):
            raise TypeError(f"装饰器 {decorator.__name__} 只能应用于BaseModel的子类上")
        return decorator(cls)
    return wrapper

