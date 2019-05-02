# Siraj Raval - Learn Python for Data Science 1
from sklearn import tree,svm

#[height, weight, shoe size]
X = [[181,80,43],[177,50,39],[160,60,38],[154,54,37],
     [166,65,40],[190,90,47],[175,64,39],[177,70,40],
     [159,60,40],[171,75,40],[181,85,43],[170,75,40]]
y = ['male','female','female','female',
     'male','male','male','female',
     'male','female','male','male']

# decision tree classifier
# clf1 = tree.DecisionTreeClassifier()
clf1 = svm.SVC(gamma='scale')

clf1 = clf1.fit(X,y)

prediction1 = clf1.predict([[190,85,43]])
# predprob1 = clf1.predict_proba([[190,85,43]])

prediction2 = clf1.predict([[160,60,37]])
# predprob2 = clf1.predict_proba([[160,60,37]])

print("Prediction 1: %s" % prediction1)
# print("Prediction 1 prob: %s" % predprob1)

print("Prediction 2: %s" % prediction2)
# print("Prediction 2 prob: %s" % predprob2)

print("Score: %s" % clf1.score(X,y))

# another classifier
# clf2 = svm.SVC(gamma='scale')

# clf2 = clf2.fit(X,y)

# prediction2 = clf2.predict([[190,70,43]])

# print("Prediction 2: ")
# print(prediction2)

# another classifier
# clf3 = tree.DecisionTreeClassifier()

# clf3 = clf3.fit(X,y)

# prediction3 = clf3.predict([[190,70,43]])

# print('Prediction 3 ' + prediction3)

