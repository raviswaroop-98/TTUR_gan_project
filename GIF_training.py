#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import imageio
import re
import pandas as pd
anim_file = 'gan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('Epoch_*.png')
    df=pd.DataFrame({'filenames':filenames})
    df['num']=df.filenames.apply(lambda x: re.findall(r"[0-9]+", x)[0]).astype(int)
    df=df.sort_values(by=['num'])
    filenames=df.filenames
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


# In[ ]:




