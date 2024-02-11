# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import gpt_decoder
import qdrant_encoder

LOGGER = get_logger(__name__)

def display(results, data):
    hotel = results
    text = data

    for x in range(len(hotel)):
        hotelName = hotel[x]['hotel_name']
        hotelDescription = hotel[x]['hotel_description']
        hotelImage = hotel[x]['hotel_image']
        priceRange = hotel[x]['price_range']
        ratingValue = hotel[x]['rating_value']
        reviewCount = hotel[x]['review_count']
        hotelUrl = hotel[x]['hotel_url']
        reason = gpt_decoder.generateText(text)
        

        with st.container(border=True):
            col1, col2 = st.columns([0.6,0.4])
            with col1:
                st.markdown(hotelName)
                st.markdown("**Price:** {} |  **Rating:** {}:star: | {:,} reviews".format(priceRange, ratingValue, reviewCount))
                st.markdown(hotelDescription)
                st.markdown("Why we chose this for you: " + reason)
                st.page_link(hotelUrl, label = "***Interested?***")
            with col2:
                st.image(hotelImage)

def run():
    st.set_page_config(
        page_title="Hotels",
    )

    with st.container(border=True):
      st.markdown('<h1 style="text-align: center; color: #89CFF0; font-family: sans-serif;">Hotel Finder</h1>', unsafe_allow_html=True)

    data = st.text_input('Search for a hotel or enter what you are looking for:',placeholder='Search...')

    results = qdrant_encoder.main(data)

    if results == []:
        st.error("Hmm.. no results found. Search again?")
    else:
        display(results, data)


if __name__ == "__main__":
    run()