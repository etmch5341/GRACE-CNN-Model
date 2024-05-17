import numpy as np

NPZPath = '/Users/etch5/Desktop/GRACE Python/'
NPZFname = 'SA_monthly_trainingTesting_numpy.npz'
imageFname = 'SA_AOHISMonthlyTraining_GRACEmonthlyTest_SEESArch_numpy_052021.png'
'''
with np.load(NPZPath+NPZFname) as data:
	
	fileTest = data['trainingData']
	print(fileTest)
	print(fileTest.shape)
	fileTest = np.squeeze(fileTest)
	print(fileTest)
	print(fileTest.shape)


	#np.savetxt('/Users/etch5/Desktop/GRACE Python/matrixData.txt', fileTest, delimiter = ' ')
'''


#import numpy as np

# Generate some test data

with np.load(NPZPath+NPZFname) as data:
	fileTest = data['trainingData']
	# Write the array to disk
	with open('test.txt', 'w') as outfile:
	    # I'm writing a header here just for the sake of readability
	    # Any line starting with "#" will be ignored by numpy.loadtxt
	    outfile.write('# Array shape: {0}\n'.format(fileTest.shape))
	    
	    # Iterating through a ndimensional array produces slices along
	    # the last axis. This is equivalent to data[i,:,:] in this case
	    for data_slice in fileTest:

	        # The formatting string indicates that I'm writing out
	        # the values in left-justified columns 7 characters in width
	        # with 2 decimal places.  
	        np.savetxt(outfile, data_slice, fmt='%-7.2f')

	        # Writing out a break to indicate different slices...
	        outfile.write('# New slice\n')