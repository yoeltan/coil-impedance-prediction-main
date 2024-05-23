import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)


# def create_table():
#     return
    # with st.connection('history_db', type='sql').session as s:
    #     s.execute('''
    #         CREATE TABLE IF NOT EXISTS history (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             predict_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    #             pid_lv INTEGER,
    #             lid_lv INTEGER,
    #             tid_lv INTEGER,
    #             pod_lv INTEGER,
    #             lod_lv INTEGER,
    #             tod_lv INTEGER,
    #             pid_hv INTEGER,
    #             lid_hv INTEGER,
    #             tid_hv INTEGER,
    #             pod_hv INTEGER,
    #             lod_hv INTEGER,
    #             tod_hv INTEGER,
    #             impedance REAL
    #         )
    #     ''')
    #     s.commit()

# create_table()