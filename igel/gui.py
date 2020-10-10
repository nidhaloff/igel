import os
import platform
import tkinter as tk
from tkinter.filedialog import askopenfilename

# really shitty function to get the readme as stantard ascii
def readme():
    try:
        import docutils
    except ImportError:
        try:
            with open(os.path.join(os.path.dirname(__file__), '../README.rst')) as f:
                return f.read()
        except (IOError, OSError):
            return ''
    with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as f:
        document = docutils.core.publish_doctree(f.read())
        nodes = list(document)
        description = ''
        for node in nodes:
            if str(node).startswith('<topic classes="contents"'):
                break
            if type(node) is docutils.nodes.comment\
            or type(node) is docutils.nodes.title:
                continue
            description += node.astext() + '\n'
        return description.encode('ascii', 'ignore').strip()

def entry_str(name,col,row,set_def=None):
    tk.Label(root, text=name).grid(column=col,row=row)
    var = tk.StringVar()
    var.set(set_def)
    tk.Entry(root,textvariable=var).grid(column=col+1,row=row)
    return var

def entry_int(name,col,row,set_def=None):
    tk.Label(root, text=name).grid(column=col,row=row)
    var = tk.IntVar()
    var.set(set_def)
    tk.Entry(root,textvariable=var).grid(column=col+1,row=row)
    return var

def optionmenu(name,col,row,options):
    tk.Label(root, text=name).grid(column=col,row=row)
    var = tk.StringVar(root)
    var.set(options[0])
    tk.OptionMenu(root,var,*options).grid(column=col+1,row=row)
    return var

def checkbox_bool(name,col,row):
    tk.Label(root, text=name).grid(column=col,row=row)
    var = tk.BooleanVar()
    var.set(False)
    tk.Checkbutton(root,text='True',variable=var).grid(column=col+1,row=row)
    return var

def opvars():
    global opvars_rowcount
    global opvars_complete
    op = opvar_var.get()
    tk.Label(root,text=op).grid(column=0,row=opvars_rowcount)
    var = tk.StringVar()
    tk.Entry(root,textvariable=var).grid(column=1,row=opvars_rowcount)
    opvars_rowcount +=1
    opvars_complete.append([op,var])
    pass

def param_grid_add_row():
    global param_rowcount
    global param_complete
    global param_grid_no_cols
    params_list = []
    for col in range(param_grid_no_cols.get()):
        var = tk.StringVar()
        tk.Entry(root,textvariable=var).grid(column=5+col,row=param_rowcount)
        params_list.append(var)
    param_complete.append(params_list)
    param_rowcount+=1
    pass

def get_model_type():
    global model_type
    global model_type_var
    model_type_var = model_type.get()

def main():
    global root
    root = tk.Tk()
    root.title("Igel GUI")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # set root directory for files, check if windows and pull up filesystem
    rootdir = '/'
    if platform.system()=='Windows':
        rootdir='C:\\'
    # filename = askopenfilename(initialdir=rootdir)
    filename = askopenfilename()
    tk.Label(root, text='Chosen File: %s'%filename).grid(column=0,row=0)
    # get filetype
    filetype = filename.split('/')[-1].split('.')[-1]
    tk.Label(root, text='File Type: %s'%filetype).grid(column=0,row=1)

    # do the whole read_data_options nonsense
    textbase = readme().split('\n')
    read_data_options = []
    for line in textbase[276:314]:
        read_data_options.append(line.split(':')[0].strip())
    tk.Label(root, text='Read Data Options:').grid(column=0,row=2)

# i'm sorry this is a huge fucking mess
    global opvar_var
    opvar_var = tk.StringVar(root)
    opvar_var.set(read_data_options[0])
    tk.OptionMenu(root,opvar_var,*read_data_options).grid(column=1,row=2)
    global opvars_rowcount
    opvars_rowcount = 3
    tk.Button(root, text='Add Option',command=opvars).grid(column=2,row=2)
    global opvars_complete
    opvars_complete = []

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tk.Label(root, text='Split Options').grid(column=4,row=0)

    split_test_size = entry_int('Test Size',4,1,0.2)

    split_shuffle = checkbox_bool('Shuffle',4,2)

    split_stratify = entry_str('Stratify (Seperate by comma)',4,3)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tk.Label(root, text='Preprocess').grid(column=4,row=4)

    preprocess_missing_values_list = ['mean','median','most_frequent','constant']
    preprocess_missing_values = optionmenu('Missing Values',4,5,preprocess_missing_values_list)

    preprocess_encoding_type_list = ['oneHotEncoding','labelEncoding']
    preprocess_encoding_type = optionmenu('Encoding Type',4,6,preprocess_encoding_type_list)

    preprocess_scale_method_list = ['standard','minmax']
    preprocess_scale_method = optionmenu('Scaling Method',4,7,preprocess_scale_method_list)

    preprocess_scale_target_list = ['inputs','outputs','all']
    preprocess_scale_target = optionmenu('Scaling Target',4,8,preprocess_scale_target_list)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tk.Label(root, text='Model Definition').grid(column=4,row=9)

    model_type_list = ['regression','classification','clustering']
    global model_type
    model_type = optionmenu('Model Type',4,10,model_type_list)
    # tk.Button(root, text='Confirm Type',command=get_model_type).grid(column=6,row=10)
    algo_options_regression = []
    algo_options_classification = []
    algo_options_clustering = []
    for line in textbase[107:134]:
        temp_algo_list = []
        for i in line.split('|'):
            temp_algo_list.append(i)
        temp_algo_list = [i.strip() for i in temp_algo_list[1:4]]
        algo_options_regression.append(temp_algo_list[0])
        algo_options_classification.append(temp_algo_list[1])
        algo_options_clustering.append(temp_algo_list[2])
    global model_type_var
    model_type_var = 'regression'
    if model_type_var=='regression':
        algo_options=algo_options_regression
    elif model_type_var=='classification':
        algo_options=algo_options_classification
    elif model_type_var=='clustering':
        algo_options=algo_options_clustering
    model_algorithm = optionmenu('Algorithm',4,11,algo_options)

    model_arguments = entry_str('Arguments (Seperate by comma)',4,12)

    model_use_cv_estimator = checkbox_bool('Use CV Estimator',4,13)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tk.Label(root, text='Cross Validate').grid(column=4,row=14)

    model_cross_validate_cv = entry_int('CV',4,15,5)

    model_cross_validate_n_jobs = entry_int('N Jobs',4,16)

    model_cross_validate_verbose = entry_int('Verbose',4,17,0)

    tk.Label(root,text='Hyperparameter Search').grid(column=4,row=18)

    model_hyperparameter_search_method_list = ['grid_search','random_search']
    model_hyperparameter_search_method = optionmenu('Method',4,19,model_hyperparameter_search_method_list)

    tk.Label(root, text='Parameter Grid').grid(column=4,row=20)
    global param_grid_no_cols
    param_grid_no_cols = entry_int('Number of Columns',4,21,2)
    global param_rowcount
    global param_complete
    param_complete = []
    param_rowcount = 22
    tk.Button(root, text='Add Grid Row',command=param_grid_add_row).grid(column=6,row=21)

    model_hyperparameter_search_arguments = entry_str('Arguments (Seperate by comma)',6,0)

    model_hyperparameter_search_arguments_cv = entry_int('CV',6,1,5)

    model_hyperparameter_search_arguments_refit = checkbox_bool('Refit',6,2)

    model_hyperparameter_search_arguments_return_train_score = checkbox_bool('Return Train Score',6,3)

    model_hyperparameter_search_arguments_verbose = entry_int('Verbose',6,4,0)

    target = entry_str('Target (Seperate by comma)',6,5)

    root.mainloop()

    opvars_dict = {}
    for x in opvars_complete:
        opvars_dict[x[0]] = x[1].get()

    parameter_grid_dict = {}
    param_num = 1
    for x in param_complete:
        param_getter_list = []
        for i in x:
            param_getter_list.append(i.get())
        param_num_str = 'param%d'%param_num
        parameter_grid_dict[param_num_str] = param_getter_list
        param_num+=1

    yaml_dict = {
        'dataset':[filename, {
            'type':filetype,
            'read_data_options':opvars_dict,
            'split': {
                'test_size':split_test_size.get(),
                'shuffle':split_shuffle.get(),
                'stratify':split_stratify.get().split(',')
            },
            'preprocess': {
                'missing_values': preprocess_missing_values.get(),
                'encoding': {
                    'type':preprocess_encoding_type.get()
                },
                'scale': {
                    'method':preprocess_scale_method.get(),
                    'target':preprocess_scale_target.get()
                }
            }
        }],
        'model': {
            'type': model_type.get(),
            'algorithm':model_algorithm.get(),
            'arguments':model_arguments.get().split(','),
            'use_cv_estimator':model_use_cv_estimator.get(),
            'cross_validate': {
                'cv':model_cross_validate_cv.get(),
                'n_jobs':model_cross_validate_n_jobs.get(),
                'verbose':model_cross_validate_verbose.get()
            },
            'hyperparameter_search': {
                'method':model_hyperparameter_search_method.get(),
                'parameter_grid':parameter_grid_dict,
                'arguments':[model_hyperparameter_search_arguments.get().split(','),{
                    'cv':model_hyperparameter_search_arguments_cv.get(),
                    'refit':model_hyperparameter_search_arguments_refit.get(),
                    'return_train_score':model_hyperparameter_search_arguments_return_train_score.get(),
                    'verbose':model_hyperparameter_search_arguments_verbose.get()
                }]
            }

        },
        'target': target.get().split(',')
    }

    print(yaml_dict)


if __name__ == '__main__':
    main()
