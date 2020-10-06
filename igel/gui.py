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

def opvars():
    global rowcount
    global complete
    op = opvar.get()
    tk.Label(root,text=op).grid(column=0,row=rowcount)
    tk.Entry(root).grid(column=1,row=rowcount)
    rowcount +=1

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
    global opvar
    opvar = tk.StringVar(root)
    opvar.set(read_data_options[0])
    tk.OptionMenu(root,opvar,*read_data_options).grid(column=1,row=2)
    global rowcount
    rowcount = 3
    tk.Button(root, text='Add Option',command=opvars).grid(column=2,row=2)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tk.Label(root, text='Split Options').grid(column=4,row=0)

    tk.Label(root, text='Test Size').grid(column=4,row=1)
    testsize_var = tk.StringVar()
    tk.Entry(root,textvariable=testsize_var).grid(column=5,row=1)

    tk.Label(root, text='Shuffle').grid(column=4,row=2)
    shuffle_var = tk.BooleanVar()
    shuffle_var.set(False)
    tk.Checkbutton(root,text='True',variable=shuffle_var).grid(column=5,row=2)

    tk.Label(root, text='Stratify (seperate by comma)').grid(column=4,row=3)
    stratify_var = tk.StringVar()
    tk.Entry(root,textvariable=stratify_var).grid(column=5,row=3)

    tk.Label(root, text='Preprocess').grid(column=4,row=4)

    tk.Label(root, text='Encoding').grid(column=4,row=5)
    encoding_var = tk.StringVar(root)
    encoding_var.set('oneHotEncoding')
    tk.OptionMenu(root,encoding_var,'oneHotEncoding','labelEncoding').grid(column=5,row=5)

    tk.Label(root, text='Scaling Method').grid(column=4,row=6)
    scalemodel_var = tk.StringVar(root)
    scalemodel_var.set('standard')
    tk.OptionMenu(root,scalemodel_var,'standard','minmax').grid(column=5,row=6)

    tk.Label(root, text='Scaling Target').grid(column=4,row=7)
    scaletarget_var = tk.StringVar(root)
    scaletarget_var.set('inputs')
    tk.OptionMenu(root,scaletarget_var,'inputs','outputs','all').grid(column=5,row=7)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tk.Label(root, text='Model Definition').grid(column=6,row=0)

    tk.Label(root, text='Model Type').grid(column=6,row=1)
    modeltype_var = tk.StringVar(root)
    modeltype_var.set('regression')
    tk.OptionMenu(root,modeltype_var,'regression','classification','clustering').grid(column=7,row=1)

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
    tk.Label(root, text='Algorithm').grid(column=6,row=2)
    algo_var = tk.StringVar(root)
    algo_var.set(algo_options[0])
    if modeltype_var.get()=='regression':
        algo_options=algo_options_regression
    elif modeltype_var.get()=='classification':
        algo_options=algo_options_classification
    elif modeltype_var.get()=='clustering':
        algo_options=algo_options_clustering
    tk.OptionMenu(root,algo_var,*algo_options).grid(column=7,row=2)


    root.mainloop()


    opvars_dict = {}

    yaml_dict = {
    'dataset':filename,
    'type':filetype,
    'read_data_options':'other dict',
    'split_test_size':testsize_var.get(),
    'split_shuffle':shuffle_var.get(),
    'split_stratify':stratify_var.get().split(','),
    'preprocess_missing_values': 'get',
    'preprocess_encoding_tpye': encoding_var.get(),
    'preprocess_scale_method':scalemodel_var.get(),
    'preprocess_scale_target':scaletarget_var.get()
    }

    print(yaml_dict)


if __name__ == '__main__':
    main()
