import os
import tkinter as tk

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


def text_field(widget_label,widget_col,widget_row,entry_width):
    text_label = tk.Label(window,text = widget_label)
    text_label.grid(column=widget_col,row=widget_row)
    text_entry = tk.Entry(window,width=entry_width)
    text_entry.grid(column=widget_col+1,row=widget_row)

def dropdown(widget_label,widget_col,widget_row,options):
    text_label = tk.Label(window,text = widget_label)
    text_label.grid(column=widget_col,row=widget_row)
    variable = tk.StringVar(window)
    variable.set('Options')
    dropdown_widget = tk.OptionMenu(window,variable,*options)
    dropdown_widget.grid(column=widget_col+1,row=widget_row)

def main():
    textbase = readme().split('\n')

    # maybe this would be better as a file selector
    db_text_field = text_field('Database',0,0,20)
    type_text_field = text_field('Type',0,1,20)
    read_data_options = []
    for line in textbase[276:314]:
        read_data_options.append(line.split(':')[0].strip())
    # here i'll put a dropdown with read_data_options and a button to add one
    # then it will create a new row with the name of the option and a textbox
    data_dropdown = dropdown('Read Data Options',0,2,read_data_options)

    window.mainloop()



if __name__ == '__main__':
    window = tk.Tk()
    window.title("Igel GUI")
    main()
