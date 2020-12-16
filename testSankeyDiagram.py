from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.graph_objects as go

source = [1,1,1,2,3,3,4,4,5,5,6]
      

target = [5,6,4,4,6,4,7,9,8,9,9]

value =  [9,3,6,2,4,7,5,3,2,6,5]

label = ['','excellent1', 'mixed1','weak1',  #0,1,2
         'excellent2', 'mixed2','weak2',  #3,4,5
         'excellent3', 'mixed3','weak3'   #6,7,8      
         ]

color_node = ['#808B96','#808B96', '#EC7063', '#F7DC6F', 
              '#808B96', '#EC7063', '#F7DC6F', 
              '#808B96', '#EC7063', '#F7DC6F']

# data to dict, dict to sankey
link = dict(source = source, target = target, value = value)
node = dict(label = label, pad=50, thickness=5, color=color_node)
data = go.Sankey(link = link, node=node)
# plot
fig = go.Figure(data)
fig.show()
plot(fig)
