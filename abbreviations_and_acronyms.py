import pandas as pd

ipath = "/home/mhaghigh/dissertation/"
ifile = ipath+"abrev.csv"
ofile = ipath+"abrev_list.tex"

df_abrev = pd.read_csv(ifile, header=0)

longest_entry = max(df_abrev['abrev_tex'], key=len)
position = [i for i, x in enumerate(df_abrev['abrev_tex']) if x == longest_entry][0]
longest_entry_string = (list(df_abrev['abrev']))[position]

with open(ofile, 'w') as out:
    out.write(r'\begin{acronym}['+longest_entry_string+']\itemsep2pt'+'\n')
    acro_list = []
    for row_element in zip(df_abrev['abrev_tex'], df_abrev['abrev'], df_abrev['full']):
        acro_list.append(r'\acro'+'{'+row_element[0]+'}'+'['+row_element[1]+']'+'{'+row_element[2]+'}'+'\n')
    acro_list.sort()
    for entry in acro_list:
        out.write(entry)
    out.write('\end{acronym}')

exit()