# Assignment 1

## Student Details
**Student Name:** Chaitanya Agrawal  
**Batch No.:** Batch 1

## Topics

#### Epoch
Epoch is a cycle for which the model is trained on the entire dataset. It includes one forward pass and one backward pass of all training dataset. Generally, we split the data into batches for training. Batch is a subset of data on which model is trained. Once all the batches of data is trained, one epoch is completed. So, each time the whole training data is iterated, an epoch is completed. For example, if you have 100 epochs then your model would be trained 100 times on the entire training dataset. As the number of epochs increase, the training error decrease further and further. After a certain value, training loss doesn't decrease further and may even increase. We need to stop increasing the epoch value for the network around this value.

#### Filter/Kernel
A filter (or kernel) is a matrix that is applied to the entire image so that it transforms the input values for further layers. Generally, kernel is a small size matrix in comparison to the input matrix size and it slides over the entire input image row by row summing up from dot product of the input matrix section and kernel areas into a single entry. For example, kernel of size 3 x 3 with the input of 5 x 5 involves taking sections equal to size of kernel(3 x 3) from the input image, and convolving between the values in the seciont and in the kernel matrix. Finally, generating a matrix of 3 x 3 as output. Also, the values of the kernel matrix change with each backpropagation iteration over the training data.
