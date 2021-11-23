import pandas as pd
path='mnist_results/mnist_0.3_NN_prox_100-pinit-eta100.0-feta100.0.pkl'   # pkl文件所在路径
	   
data = pd.read_pickle(path)

# print(data)

writer = pd.ExcelWriter(path[:-4]+'.xlsx')
df1 = pd.DataFrame(data)
df1.to_excel(writer,'Sheet1')
writer.save()
# print(data.columns) 
# print(data.values)

                
    