import streamlit as st
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

## pepare data
# Get a list of all JSON files in the corresponding directory
json_results = glob.glob('results/*.json')

for r_file in json_results:
    with open(r_file, 'r') as f1:
        # Use the file name (without the extension and directory path) as the list name
        list_name = os.path.splitext(os.path.basename(r_file))[0]
        locals()[list_name] = json.load(f1)
        f1.close()


result_map = {
    ('Beispiel 1', 1,'Easycut A'): bsp1_bc,
    ('Beispiel 1', 1,'Easycut B'): bsp1_pc,
    ('Beispiel 1', 2, 'Easycut A'): bsp1_dp,
    ('Beispiel 1', 2, 'Easycut B'): bsp1_dp_pc,
    ('Beispiel 1', 3, 'Easycut A'): bsp1_tp,
    ('Beispiel 1', 3, 'Easycut B'): bsp1_tp_pc,
    ('Beispiel 2', 1, 'Easycut A'): bsp2_bc,
    ('Beispiel 2', 1, 'Easycut B'): bsp2_pc,
    ('Beispiel 2', 2, 'Easycut A'): bsp2_dp,
    ('Beispiel 2', 2, 'Easycut B'): bsp2_dp_pc,
    ('Beispiel 2', 3, 'Easycut A'): bsp2_tp,
    ('Beispiel 2', 3, 'Easycut B'): bsp2_tp_pc,
    ('Beispiel 3', 1, 'Easycut A'): bsp3_bc,
    ('Beispiel 3', 1, 'Easycut B'): bsp3_pc,
    ('Beispiel 3', 2, 'Easycut A'): bsp3_dp,
    ('Beispiel 3', 2, 'Easycut B'): bsp3_dp_pc,
    ('Beispiel 3', 3, 'Easycut A'): bsp3_tp,
    ('Beispiel 3', 3, 'Easycut B'): bsp3_tp,
}

## functions

def vistrack_sp(tracks,full_profiles, cut_items,profile):
    mpl.rcParams.update({'font.size': 10})
    fig,ax = plt.subplots()
    fp = np.array(full_profiles)*profile
    trackplan =pd.DataFrame({'ganze Profile': fp,
                             'Teilstück': np.array(cut_items)[:,0],
                             'Teilstück ': np.array(cut_items)[:,1],
                             }, index =tracks)
    ax = trackplan.plot.barh(title="Schienenaufteilung", stacked = True )
    for c in ax.containers:
        labels = [
            str(int(v.get_width() / profile)) + 'x {}'.format(profile)
            if v.get_width() > profile and v.get_width() % profile == 0
            else str(int(v.get_width())) if v.get_width() > 0
            else ''
            for v in c
]
        ax.bar_label(c,labels = labels, label_type='center', fontsize = 8)
        ax.tick_params(axis='both', which='major', labelsize=8)
    fig = ax.get_figure()
    return fig


def vistrack_mp(tracks,full_profiles,cut_items,profiles):
    fig,ax = plt.subplots()
    trackplan = pd.DataFrame(full_profiles*profiles, index = tracks)
    trackplan.columns = profiles
    if cut_items.ndim == 1:
        cut_items = cut_items[:,np.newaxis]
    for i in range(cut_items.shape[1]):
        trackplan['Teilstück {}'.format(i+1)] = cut_items[:,i]
    ax = trackplan.plot.barh( title= 'Schienenaufteilung', stacked=True)
    for c in ax.containers:
        labels=[str(int(v.get_width())) if v.get_width() > 0 else '' for v in c]
        for i, profile in enumerate(profiles):
            for j, v in enumerate(c):
                if v.get_width() > profile and v.get_width() % profile == 0:
                    labels[j] = str(int(v.get_width() / profile)) + 'x {}'.format(profile)

        ax.bar_label(c, label_type='center',labels = labels,fontsize = 8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize = 6)
    fig = ax.get_figure()
    return fig

def viscut_sp(cut_profiles, profile):
    fig,ax = plt.subplots()
    cutplan = pd.DataFrame(cut_profiles)
    new_columns = [f'item {i+1}' for i in range(len(cutplan.columns)) ]
    cutplan.columns = new_columns
    colors = []  # initialze colors list
    for i, column in enumerate(cutplan.columns):
        colors.append('C{}'.format(i+4))  # C3 is red and will be used for the loss
    cutplan['loss'] = profile - cutplan.iloc[:,0:].sum(axis=1)
    colors.append('C3')
    ax = cutplan.plot.barh(title="Profilzuschnitt", stacked = True, figsize=(10,6), color = colors)
    for c in ax.containers:
        labels = [str(int(v.get_width())) + "mm" if v.get_width() > 0 else '' for v in c]
        ax.bar_label(c,labels = labels, label_type='center', fontsize = 8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize = 8)
    fig = ax.get_figure()
    return fig


def viscut_mp(cut_profiles, q_length):
    fig,ax = plt.subplots()
    cutplan = pd.DataFrame(cut_profiles).fillna(0)
    new_columns = [f'item {i+1}' for i in range(len(cutplan.columns)) ]
    cutplan.columns = new_columns
    colors = []  # initialze colors list
    for i, column in enumerate(cutplan.columns):
        colors.append('C{}'.format(i+4))  # C3 is red and will be used for the loss
    cutplan['loss'] = q_length - cutplan.iloc[:,0:].sum(axis=1)
    colors.append('C3')
    cutplan.index = np.array(q_length).astype(int)
    ax = cutplan.plot.barh(stacked = True, color = colors)
    ax.set_title("Profilzuschnitt", fontsize = 10)
    for c in ax.containers:
        labels = [str(int(v.get_width())) + "mm" if v.get_width() > 0 else '' for v in c]
        ax.bar_label(c,labels = labels, label_type='center', fontsize = 7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(fontsize = 7)
    fig = ax.get_figure()
    return fig







st.set_page_config(
    page_title="Schienenzuschnitt",
    layout="wide")

st.title("Willkommen zum Easycut-Test")

st.markdown("""

Herzlichen Dank, dass Sie sich die Zeit nehmen, unsere Ansätze für den automatischen Schienenzuschnitt zu begutachten.


Wir haben für Sie eine interaktive Website vorbereitet, in der Sie verschiedene Einstellungen ausprobieren und die Ergebnisse vergleichen können.
Nachdem Sie eines unserer Beispieldächer ausgewählt haben, können Sie noch Angaben zu den zu berücksichtigenden Profillängen machen. Anschließend
wählen Sie zwischen zwei Verfahren, die Zuschnitte zu bestimmen:

* **Easycut - A**:
    Diese Variante produziert sehr einfache Zuschnittsmuster mit einer maximalen
    Anzahl an identischen Teilstücken. Unter Umständen kann dabei der Verschnitt etwas größer ausfallen.
* **Easycut - B**:
    Auch hier werden einfache Zuschnittsmuster bestimmt, allerdings sind mehr
    unterschiedliche Teilstücke erlaubt, was den Verschnitt gegenüber Variante A verringert.


:bulb: Um andere Varianten und Kombinationen aus zu probieren, ändern Sie einfach Ihre Angaben und klicken anschließend auf Zuschnittsplan jetzt berechnen!
""")

st.subheader("Beispieldächer")

col1, col2, col3 = st.columns(3)
col1.image(Image.open('images/bsp1.png'),caption="Beispieldach 1", use_column_width=True)
col2.image(Image.open('images/bsp2.png'),caption="Beispieldach 2", use_column_width=True)
col3.image(Image.open('images/bsp3.png'),caption="Beispieldach 3", use_column_width=True)



with st.sidebar.form("parameter"):
   st.title("Parameter")
   st.subheader("Beispiel wählen")
   options_example = ['Beispiel 1', 'Beispiel 2', 'Beispiel 3']
   selection_example = st.selectbox('Für welches Dach wollen Sie einen Schienenzuschnittsplan erstellen?', options_example,key="bsp", index = 0)

   st.subheader("Profilangaben")

   p_list=[6600,6000,3600,2400]
   pchoice = st.multiselect('Markieren Sie bis zu **drei** gewünschte Profillängen', p_list, default=6600, max_selections =4)
   selection_num_prof = len(pchoice)
   st.subheader("Varianten")
   options_cut = ['Easycut A', 'Easycut B']
   selection_cut = st.radio('Welche Variante wollen sie testen?', options_cut, index = 0)

   list_key =  (selection_example,selection_num_prof,selection_cut)

   # Every form must have a submit button.
   Submit = st.form_submit_button("Zuschnittsplan jetzt berechnen")


   if Submit:

       results = result_map[list_key]
       st.session_state.submitted = True

       if selection_num_prof == 1 :
           Full_prof_dict, Cut_items_dict, Cut_prof_dict,needed_profiles_dict,total_cuts_dict,connectors_dict, loss_dict = [results[i] for i in range(len(results))]
           Profile = pchoice[0]
           Full_profiles = Full_prof_dict[str(pchoice[0])]
           Cut_items = Cut_items_dict[str(pchoice[0])]
           Cut_profiles = Cut_prof_dict[str(pchoice[0])]
           needed_profiles = needed_profiles_dict[str(pchoice[0])]
           total_cuts = total_cuts_dict[str(pchoice[0])]
           connectors = connectors_dict[str(pchoice[0])]
           loss = loss_dict[str(pchoice[0])]

       else:
           Full_prof_dict, Cut_items_dict, Cut_prof_dict, used_p_length_dict, needed_profiles_dict, total_cuts_dict, connectors_dict, loss_dict = [results[i] for i in range(len(results))]
           P_string = str(tuple(sorted(pchoice, reverse = True)))
           Full_profiles = Full_prof_dict[P_string]
           Cut_items = np.array(Cut_items_dict[P_string])
           Cut_profiles = Cut_prof_dict[P_string]
           used_p_length = used_p_length_dict[P_string]
           needed_profiles = needed_profiles_dict[P_string]
           total_cuts = int(total_cuts_dict[P_string])
           connectors = connectors_dict[P_string]
           loss = loss_dict[P_string]
           Profiles = np.array(sorted(pchoice,reverse =True))


if st.session_state['submitted']==True:

    data_map ={'Beispiel 1': 'bsp1', 'Beispiel 2': 'bsp2', 'Beispiel 3': 'bsp3'}
    example = data_map[selection_example]

    # Load the JSON file and extract its contents
    json_path = f"data/{example}.json"
    with open(json_path, 'r') as f2:
            data = json.load(f2)

    Tracks = np.array(data['tracks'])  # track lengths
    l_min = np.array(data['rafter_dist']) + np.array(data['overhang']) # Minimum Item length for every track #

    subheader_text= f"Zuschnittsplan für {selection_example} und Profillänge(n) {pchoice} mit Variante {selection_cut}"
    st.subheader(subheader_text)
    st.markdown(""" # """)
    table1, plot1 = st.columns((2,3))
    table2, plot2 =st.columns((1,2))
    if selection_num_prof == 1:
        df1 = pd.DataFrame({'Schienenlänge': Tracks, 'Überhang': data['overhang'], 'Mindestlänge': l_min})
        df1 = df1.iloc[::-1]
        track_df= df1.style.hide()
        st.markdown(""" # """)
        table1.write(track_df.to_html(), unsafe_allow_html=True)
        with plot1:
            st.pyplot(vistrack_sp(Tracks,Full_profiles, Cut_items,Profile))

        cut_data = {f"benötigte Profile mit Länge {Profile}": needed_profiles, 'Anzahl Schnitte': total_cuts, 'Verbinder': connectors, 'Gesamtverschnitt':loss}
        df2 = pd.DataFrame(cut_data.items())
        cut_df = df2.style.hide()
        cut_df.hide_columns()
        table2.write(cut_df.to_html(), unsafe_allow_html=True)
        with plot2:
            st.pyplot(viscut_sp(Cut_profiles, Profile))
    else:
        df1 = pd.DataFrame({'Schienenlänge': Tracks, 'Überhang': data['overhang'], 'Mindestlänge': l_min})
        df1 = df1.iloc[::-1]
        track_df = df1.style.hide()
        table1.write(track_df.to_html(), unsafe_allow_html=True)
        with plot1:
            st.pyplot(vistrack_mp(Tracks,Full_profiles, Cut_items,Profiles))

        p_data={}
        for i,Profile in enumerate(Profiles):
            p_data[f"Profile mit Länge {Profile}"] = needed_profiles[i]
        cut_data = {'Anzahl Schnitte': total_cuts, 'Verbinder': int(connectors), 'Gesamtverschnitt':loss}
        p_data.update(cut_data)
        df2 = pd.DataFrame(p_data.items())
        cut_df = df2.style.hide()
        cut_df.hide_columns()
        table2.write(cut_df.to_html(), unsafe_allow_html=True)
        with plot2:
            st.pyplot(viscut_mp(Cut_profiles, used_p_length))
else :
    st.markdown("""
    Wählen Sie die gewünschten Einstellungen in dem Formular in der linken Spalte und klicken Sie
    auf Zuschnittsplan erstellen
    """)
