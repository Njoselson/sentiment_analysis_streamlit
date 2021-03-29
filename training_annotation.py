import streamlit as st
import pandas as pd

categories = {"good": 3, "ambiguous": 2, "skip": 1, "bad": 0}

@st.cache(allow_output_mutation=True)
def get_data():
    data = pd.read_csv("test.csv")
    data["annotation"] = None
    return data

@st.cache(allow_output_mutation=True)
def get_annotation():
    return {"row": 0}

row = st.empty()
match = st.empty()
buttons = {}

data = get_data()
annotation = get_annotation()

def detail():
    current_obs = data.loc[annotation["row"]]
    row.markdown(f"# {annotation['row'] + 1}")
    match.markdown(f"**{current_obs['location']}** matched **{current_obs['area']}**")

if annotation["row"] < len(data.index):
    for cat in categories.keys():
        buttons[cat] = st.button(cat)
    detail()
    for cat in categories.keys():
        if buttons[cat]:
            data.loc[annotation["row"], "annotation"] = categories[cat]
            annotation["row"] += 1
            if annotation["row"] < len(data.index):
                detail()
else:
    data.to_csv("test_annotated.csv")
    st.write("finished")