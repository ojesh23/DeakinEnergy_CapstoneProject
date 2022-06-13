# import libraries
from io import BytesIO
import xlsxwriter
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import os
import base64
import time
import texthero as hero
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

try:
    from pyngrok import ngrok
    #public_url = ngrok.connect('8501')
    #print(public_url)
except ImportError:
    pass
######################################

# current directory and paths
DIRECTORY = os.getcwd()
images = f"{DIRECTORY},Data App,Assets,Images".split(",")
datasets = f"{DIRECTORY},Data App,Assets,Datasets".split(",")
image_path = os.path.join(*images)
dataset_path = os.path.join(*datasets)
token = f"{DIRECTORY},Data App,Assets,.mapbox_token".split(",")
token_path = os.path.join(*token)
deakin_energy_logo = os.path.join(image_path, 'DeakinEnergy.jpg')

# app setup
st.set_page_config(page_title="Deakin Energy", page_icon=deakin_energy_logo, layout='wide', initial_sidebar_state='auto')

# display ESV logo
st.image(deakin_energy_logo, width=100)

# App title
st.title("__Deakin Energy - Data Analysis Tool__")

# display this in sidebar
st.sidebar.warning("Please select an option!")

# options in sidebar
my_page = st.sidebar.radio('Page Navigation', ['Data Analysis', 'K-Means'])

def main():
    # load dataset
    @st.cache(allow_output_mutation=True, persist=True, suppress_st_warning=True)
    def dataframe():
        df = pd.read_csv(os.path.join(dataset_path, 'cleaned_span_inspections.csv'), low_memory=False)
        return df
    df = dataframe()

    # plot map
    @st.cache
    def map_plot(df):
        # token access
        px.set_mapbox_access_token(open(token_path).read())

        # plot datapoints in map
        fig = px.scatter_mapbox(df, lat=df['Lat'], lon=df['Long'], hover_name="Address", zoom=8, height=800, width=1500,
                                hover_data=['FinancialYear', 'AdditionalInformation', 'ProgramType', 'Postcode',
                                            'Locality', 'WeatherStation', 'VegetationSpan', 'NetworkType'],
                                color='Postcode')
        return fig

    # check if page is data analysis
    if my_page == 'Data Analysis':
        with st.beta_container():
            st.subheader("__Line Inspection Data__")
            show = st.selectbox(label='Show', options = [10,50,100,500,'All'])
            if show == 'All':
                st.dataframe(df)
            else:
                st.dataframe(df[:show])
            st.success('Dataset Successfully Loaded!')

        # subheader and text to print
        st.subheader("__Non-Compliant Profiling__")
        st.markdown('__Non compliant Line Inspection Data__')

        # filters
        st.markdown("**Please select from following items to filter data**")

        is_complaint = st.selectbox('Non-Complaint', ['Yes', 'No'])

        # filter non compliant data
        @st.cache(allow_output_mutation=True)
        def filter_noncompliant(df, value):
            if is_complaint == 'Yes':
                # select only non-compliant data
                non_compliant = df[df['NonCompliant'] == 'yes'].reset_index()
                non_compliant.style.hide_index()
            else:
                non_compliant = df
            return non_compliant

        non_compliant = filter_noncompliant(df, is_complaint)

        # 3x3 filter widgets
        x1, x2, x3 = st.beta_columns(3)
        y1, y2, y3 = st.beta_columns(3)

        # filter by postcode
        postcode = x1.multiselect('Postcode', np.unique(non_compliant['Postcode']))
        if postcode != []:
            selected_df = non_compliant[non_compliant['Postcode'].isin([int(x) for x in postcode])]

        # filter by financial year
        financialYear = x2.multiselect('Financial Year', np.unique(non_compliant['FinancialYear']))
        if financialYear != []:
            selected_df = non_compliant[non_compliant['FinancialYear'].isin([x for x in financialYear])]

        # filter by vegetation span
        vegetationSpan = x3.multiselect('Vegetation Span', np.unique(non_compliant['VegetationSpan']))
        if vegetationSpan != []:
            selected_df = non_compliant[non_compliant['VegetationSpan'].isin([x for x in vegetationSpan])]

        # filter by weather station
        weatherstation = y1.multiselect('Weather Station', np.unique(non_compliant['WeatherStation']))
        if weatherstation != []:
            selected_df = non_compliant[non_compliant['WeatherStation'].isin([x for x in weatherstation])]

        # filter by locality
        Locality = y2.multiselect('Locality', np.unique(non_compliant['Locality']))
        if Locality != []:
            selected_df = non_compliant[non_compliant['Locality'].isin([x for x in Locality])]

        # filter by program type
        programType = y3.multiselect('Program Type', np.unique(non_compliant['ProgramType']))
        if Locality != []:
            selected_df = non_compliant[non_compliant['ProgramType'].isin([x for x in programType])]

        # check if no filters are selected
        if postcode==[] and financialYear==[] and weatherstation==[] and Locality==[]\
                and vegetationSpan==[] and programType==[]:
            selected_df = non_compliant

        try:
            selected_df.drop(columns=['index'], inplace=True)
        except:
            pass

        # display dataframe of non-compliant data only
        st.dataframe(selected_df)
        st.success('Dataset Successfully Filtered!')

        # download dataset
        @st.cache()
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='ESV Line Inspection')
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        @st.cache()
        def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            val = to_excel(df)
            b64 = base64.b64encode(val)
            return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="filtered_data.csv">Download csv file</a>'

        st.markdown(get_table_download_link(selected_df), unsafe_allow_html=True)

        # plot hexabin
        @st.cache()
        def hexbin_plot(df):
            fig = ff.create_hexbin_mapbox(data_frame=df, lat="Lat", lon="Long", nx_hexagon=20, opacity=0.5,
                labels={"color": "Point Count"}, min_count=1, color_continuous_scale="Viridis", show_original_data=True,
                original_data_marker=dict(size=4, opacity=0.6, color="deeppink"), height=800, width=1500 )
            return fig

        # display map and labels
        st.subheader("Non-Compliant Addresses on Map")
        st.markdown("**Please select from following items to filter data**")
        st.plotly_chart(map_plot(selected_df))

        # hexbin
        st.subheader("Non-Compliant Addresses on Hexbin Map")
        st.markdown("**A hexbin map refers to two different concepts. It can be based on a geospatial object "
                    "where all regions of the map are represented as hexagons. Or it can refer to a 2D density technique.**")
        st.plotly_chart(hexbin_plot(selected_df))

        # non-compliant dataset stats
        st.subheader("Non-Compliant Statastics")
        @st.cache()
        def stats(df, value):
            group_data = df.groupby(value)
            group_data = group_data.size().reset_index(name='counts').sort_values(by='counts', ascending=False)
            return group_data

        # divide widgets into 2 cols
        x1, x2 = st.beta_columns(2)

        # widget to select group by options
        groupby_value = x1.selectbox('Group By', ['FinancialYear', 'Postcode', 'WeatherStation', 'Locality', 'ProgramType'])

        # widget to select non-complaint
        check_complaint = x2.selectbox('Is Non-Complaint', ['Yes', 'No'])

        # conditions
        if check_complaint == 'Yes':
            # select only non-compliant data
            non_compliant = df[df['NonCompliant'] == 'yes'].reset_index()
        else:
            non_compliant = df

        # split widgets into two columsn
        x1, x2 = st.beta_columns(2)

        # display dataframe
        x1.dataframe(stats(non_compliant, groupby_value))

        # display chart
        x2.bar_chart(stats(non_compliant, groupby_value))

    elif my_page=='K-Means':
        st.title('K-Means Clustering')
        st.markdown('__Non-compliant clusters__')

        # plot map
        @st.cache
        def map_plot(df):
            # token access
            px.set_mapbox_access_token(open(token_path).read())

            # plot datapoints in map
            fig = px.scatter_mapbox(df, lat=df['Lat'], lon=df['Long'], hover_name="Address", zoom=8, height=800,
                                    width=1500,
                                    hover_data=['FinancialYear', 'AdditionalInformation', 'ProgramType', 'Postcode',
                                                'Locality', 'WeatherStation', 'VegetationSpan', 'NetworkType'],
                                    color='KmeansLabel')
            return fig

        # optimal value of K
        #st.markdown('__Optimal Value of K in K-Means__')
        #st.image(os.path.join(image_path, 'Kmeans_Kvalues.png'))

        # read kmeans cluster data
        df = pd.read_csv(os.path.join(dataset_path, 'Kmeans_data.csv'), low_memory=False)

        # show dataframe
        # filter by cluster
        cluster = st.multiselect('Cluster No', np.unique(df['KmeansLabel']))
        if cluster != []:
            df_filtered = df[df['KmeansLabel'].isin([x for x in cluster])]
        else:
            df_filtered = df

        st.dataframe(df_filtered)
        st.success('Dataset Successfully Loaded!')

        # plot
        st.markdown('__Clustering Visualization__')
        st.plotly_chart(map_plot(df_filtered))

        # plot bar chart of bin size
        st.markdown('__Number of data points in each cluster__')
        st.bar_chart(np.bincount(df['KmeansLabel']))


        # elbow chart
        st.markdown('__Clustering Evaluation__')
        st.write('The Silhouette Coefficient is used when the ground-truth about the dataset is unknown and computes '
                 'the density of clusters computed by the model. The score is computed by averaging the silhouette '
                 'coefficient for each sample, computed as the difference between the average intra-cluster distance '
                 'and the mean nearest-cluster distance for each sample, normalized by the maximum value. This '
                 'produces a score between 1 and -1, where 1 is highly dense clusters and -1 is completely incorrect '
                 'clustering. '
                 'The Silhouette Visualizer displays the silhouette coefficient for each sample on a per-cluster '
                 'basis, visualizing which clusters are dense and which are not. This is particularly useful for '
                 'determining cluster imbalance, or for selecting a value for 𝐾 by comparing multiple visualizers.')

        # split widgets into two columsn
        x1, x2 = st.beta_columns(2)

        x1.image(os.path.join(image_path, 'KMeans Elbow Plot.png'))
        x2.image(os.path.join(image_path, 'Kmeans Silhouette Plot.png'))

if __name__ == '__main__':
    main()
    st.stop()