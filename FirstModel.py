import FirstModelObject

net = FirstModelObject.Network([784, 30,10])

net.SGD(training_data, 30,10,3.0,test_data = test_data)