import toml


def params_null_replace(params_dict, null_keyword="none"):
    for key in params_dict:
        if isinstance(params_dict[key], dict):
            for sub_key in params_dict[key]:
                if isinstance(params_dict[key][sub_key], dict):
                    for sub_sub_key in params_dict[key][sub_key]:
                        if isinstance(params_dict[key][sub_key][sub_sub_key], list):
                            for sub_sub_item in range(len(params_dict[key][sub_key][sub_sub_key])):
                                if params_dict[key][sub_key][sub_sub_key][sub_sub_item] == null_keyword:
                                    params_dict[key][sub_key][sub_sub_key][sub_sub_item] = None
                        elif params_dict[key][sub_key][sub_sub_key] == null_keyword:
                            params_dict[key][sub_key][sub_sub_key] = None
                else:
                    if isinstance(params_dict[key][sub_key], list):
                        for sub_item in range(len(params_dict[key][sub_key])):
                            if params_dict[key][sub_key][sub_item] == null_keyword:
                                params_dict[key][sub_key][sub_item] = None
                    elif params_dict[key][sub_key] == null_keyword:
                        params_dict[key][sub_key] = None
        else:
            if isinstance(params_dict[key], list):
                for item in range(len(params_dict[key])):
                    if params_dict[key][item] == null_keyword:
                        params_dict[key][item] = None
            elif params_dict[key] == null_keyword:
                params_dict[key] = None

    return params_dict


def parse_params(params_path):
    params = toml.load(params_path)
    return params_null_replace(params)
