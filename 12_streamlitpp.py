import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
st.balloons()
url = 'laptops_data.csv'
data = pd.read_csv(url)

data['Price'] = data['Price'].replace('[\$,]', '', regex=True).astype(float)  
data['Rating'] = data['Rating'].astype(int)

st.title("Laptop Data Visualizations")

st.subheader('Histogram of Ratings')
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(data['Rating'], bins=range(1, 6), edgecolor='black')
ax1.set_title('Histogram of Ratings')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Frequency')
ax1.set_xticks(range(1, 6))
ax1.grid(True)
st.pyplot(fig1)

st.subheader('Scatter Plot of Price vs Rating')
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(data['Rating'], data['Price'], alpha=0.7)
ax2.set_title('Scatter Plot of Price vs Rating')
ax2.set_xlabel('Rating')
ax2.set_ylabel('Price')
ax2.grid(True)
st.pyplot(fig2)

st.subheader('Average Price by Rating')
avg_price_by_rating = data.groupby('Rating')['Price'].mean()
fig3, ax3 = plt.subplots(figsize=(10, 6))
avg_price_by_rating.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax3)
ax3.set_title('Average Price by Rating')
ax3.set_xlabel('Rating')
ax3.set_ylabel('Average Price')
ax3.set_xticks(range(len(avg_price_by_rating.index)), avg_price_by_rating.index)
ax3.grid(axis='y')
st.pyplot(fig3)

st.subheader('Distribution of Ratings')
rating_counts = data['Rating'].value_counts()
fig4, ax4 = plt.subplots(figsize=(8, 8))
ax4.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(rating_counts))))
ax4.set_title('Distribution of Ratings')
st.pyplot(fig4)

st.subheader('3D Line Plot of Device Name, Rating, and Price')
fig5 = px.line_3d(data, x="Device Name", y="Rating", z="Price")
st.plotly_chart(fig5)

st.subheader('3D Scatter Plot of Device Name, Rating, and Price')
fig6 = px.scatter_3d(data, x="Device Name", y="Rating", z="Price", color='Price', size='Rating', symbol='Rating')
st.plotly_chart(fig6)

st.subheader('Price vs Rating (Bar Plot)')
fig7 = px.bar(data, x='Rating', y="Price")
st.plotly_chart(fig7)

data['Price_Bins'] = pd.cut(data['Price'], bins=10)
st.subheader('Bar Plot with Price Bins')
fig8 = px.bar(data, x="Device Name", y="Rating", color='Price_Bins', facet_row='Rating', facet_col='Price_Bins', facet_col_spacing=0.01)
st.plotly_chart(fig8)

st.subheader('Box Plot of Price by Rating')
fig9 = px.box(data, x="Rating", y="Price")
st.plotly_chart(fig9)

st.subheader('Violin Plot of Rating vs Price')
fig10 = px.violin(data, x="Rating", y="Price")
st.plotly_chart(fig10)

st.subheader('Pie Chart of Rating Distribution by Device Name')
fig11 = px.pie(data, values="Rating", names="Device Name", color_discrete_sequence=px.colors.sequential.RdBu, opacity=0.7, hole=0.5)
st.plotly_chart(fig11)


