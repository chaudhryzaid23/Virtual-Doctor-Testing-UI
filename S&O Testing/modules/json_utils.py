def format_json_to_prompt(data, level=1):
    result = ""
    if isinstance(data, dict):
        for key, value in data.items():
            result += f"\n{'#' * level} {key}\n"
            result += format_json_to_prompt(value, level + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            result += f"\n{'#' * level} Item {i+1}\n"
            result += format_json_to_prompt(item, level + 1)
    else:
        result += f"{data}\n"
    return result 