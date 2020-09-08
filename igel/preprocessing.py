
def update_dataset_props(dataset_props: dict, default_dataset_props: dict):
    for key1 in default_dataset_props.keys():
        if key1 in dataset_props.keys():
            for key2 in default_dataset_props[key1].keys():
                if key2 in dataset_props[key1].keys():
                    default_dataset_props[key1][key2] = dataset_props[key1][key2]

    return default_dataset_props
