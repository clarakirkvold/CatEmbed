import pandas as pd
import categorical_entity_embedder as ce
import pickle

d = pd.read_csv('../train_ads.csv')
column_names = d.columns.tolist()
print(column_names)
column_names=['adsorbate', 'site', 'MetalA', 'RatioA', 'MetalB', 'RatioB', 'GPCAF_Feature1', 'GPCAF_Feature2', 'GPCAF_Feature3', 'GPCAF_Feature4', 'GPCAF_Feature5', 'GPCAF_Feature6', 'GPCAF_Feature7', 'GPCAF_Feature8', 'GPCAF_Feature9', 'GPCAF_Feature10', 'GPCAF_Feature11', 'GPCAF_Feature12', 'GPCAF_Feature13', 'GPCAF_Feature14', 'GPCAF_Feature15', 'GPCAF_Feature16', 'GPCAF_Feature17', 'GPCAF_Feature18', 'GPCAF_Feature19', 'GPCAF_Feature20']
xx =d[column_names]

yy = d['reactionEnergy']

#Preprocess
embedding_info = ce.get_embedding_info(xx)

X_encoded,encoders,labels = ce.get_label_encoded_data(xx)

#Train embedding network
embeddings = ce.get_embeddings(X_encoded, yy, categorical_embedding_info=embedding_info,
                            is_classification=False, epochs=1000)

#Transform categorical descriptors using embeddings obtained from embedding network
x_train = ce.fit_transform(X_encoded, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)

y_train = yy.to_frame()

print('Embedding complete')

#Save embeddings

with open('./embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings, file)

with open('./encoders.pkl', 'wb') as file:
    pickle.dump(encoders, file)

with open('./labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

#Save transformed x_train and y_train

x_train.to_pickle('./x_train.pkl')

y_train.to_pickle('./y_train.pkl')

print('Saved x_train and y_train')

