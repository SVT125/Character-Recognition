// Will expect images of dimensions 20 x 20 pixels (400 features)as input.
// Use file naming format e.g. "exampleL0001.jpg", where we read in the file name and take the 8th character as the output y, here "L".

// Initial neural network implementation will have 100 units per layer, 2 hidden layers.

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.analysis.function.Sigmoid;
import java.nio.file.*;
import java.io.*;
import java.util.*;
import java.awt.Color;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class NeuralNetwork {
	 List<Example> trainingExamples;
	 List<Example> testExamples;
	
	BlockRealMatrix theta1, theta2, theta3, pDerivative1, pDerivative2, pDerivative3, act2, act3;
	
	BlockRealMatrix hypothesis; // layer 4, final vector result
	BlockRealMatrix testHypothesis; // the test hypothesis when calculating the cost
	
	double learningRate; // rate multiplied by the partial derivative. Greater values mean faster convergence but possible divergence, default .01.
	double regularizationRate; // regularization rate, default .01.
	double colorSum = 0; // sum of all numbers r, g, b across all examples
	double colorNum = 0; // number of pixels across all examples
	int numExamples = 0; // number of examples
	
	int numUnits, numFeatures, numOutputs;
	
	public NeuralNetwork(String trainingFileName, String testFileName, int numUnits, int numFeatures,
		double learningRate, double regRate, int numOutputs, int numRuns) {
		this.trainingExamples = readExamples(trainingFileName, "training");
		this.testExamples = readExamples(testFileName, "test");
		this.trainingExamples = meanNormalize( trainingExamples );
		this.testExamples = meanNormalize( testExamples );
		
		this.numOutputs = numOutputs;
		this.hypothesis = new BlockRealMatrix(numOutputs,1);
		this.testHypothesis = new BlockRealMatrix(numOutputs,1);
		
		theta1 = NeuralNetwork.randInitialize( 1, numUnits + 1, numFeatures + 1 );
		System.out.println( "Randomly initialized first parameter vector." );
		theta2 = NeuralNetwork.randInitialize( 1, numUnits + 1, numUnits + 1 );
		System.out.println( "Randomly initialized second parameter vector." );
		theta3 = NeuralNetwork.randInitialize( 1, numOutputs, numUnits + 1 );
		System.out.println( "Randomly initialized third parameter vector." );
		System.out.println("---------------------");
		
		this.learningRate = learningRate; 
		this.regularizationRate = regRate;
		this.numUnits = numUnits;
		this.numFeatures = numFeatures;
		this.numOutputs = numOutputs;
		
		train(numRuns);
		
		hypothesis = forwardPropagation( testExamples.get(0));
				
	}
	
	// Runs gradient descent until the cost is less than some epsilon.
	public void train(double epsilon) {
		int iteration = 1;
		while( calculateCost() > epsilon ) {
			gradientDescent(1);
			System.out.println( "Cost of iteration " + iteration++ + " is: " + calculateCost());
		}
	}
	
	// Calculates the cost function.
	public double calculateCost() {
		double cost = 0;
		double threshold = Math.pow(Math.E,-323); // pre-calculated value for loops
		// sum over all examples
		for( int i = 0; i < numExamples; i++ ) { 
			this.testHypothesis = forwardPropagation(trainingExamples.get(i));
			// sum over all elements of the hypothesis
			for( int j = 0; j < hypothesis.getRowDimension(); j++ ) {
				double firstLogArgument = testHypothesis.getEntry(j,0), secondLogArgument = 1 - testHypothesis.getEntry(j,0);
				if( firstLogArgument < threshold )
					firstLogArgument = threshold;
				if( secondLogArgument < threshold )
					secondLogArgument = threshold;	
				cost += ( (trainingExamples.get(i).y.getEntry(j) * Math.log(firstLogArgument)) + 
					((1 - trainingExamples.get(i).y.getEntry(j)) * Math.log(secondLogArgument)) ); // define the hypothesis for each example
			}
		}
		cost = ((double)(-1)/(double) (numExamples)) * cost;
		cost += (regularizationRate/(2*numExamples)) * (NeuralNetwork.sumSquaredMatrix(theta1) + 
			NeuralNetwork.sumSquaredMatrix(theta2) + NeuralNetwork.sumSquaredMatrix(theta3));
		return cost;
	}
	
	// Runs numIterations iterations of batch gradient descent using the class' partial derivative terms calculated from backprop and the learning rate.
	public void gradientDescent( int numIterations ) {
		for( int i = 0; i < numIterations; i++ ) {
			for( Example ex : this.trainingExamples ) 
				this.hypothesis = forwardPropagation( ex );
			backPropagation( this.trainingExamples ); // updates partial derivatives
			theta1 = theta1.subtract( pDerivative1.scalarMultiply(learningRate) );
			theta2 = theta2.subtract( pDerivative2.scalarMultiply(learningRate) );
			theta3 = theta3.subtract( pDerivative3.scalarMultiply(learningRate) );
		}
	}
	
	// Runs forward propagation, given this particular neural network.
	// Can readjust to take arguments of number of units and layers.
	public BlockRealMatrix forwardPropagation(Example ex) {
		BlockRealMatrix z2 = theta1.multiply(NeuralNetwork.convertMatrix(ex.x));
		act2 = NeuralNetwork.convertMatrix(NeuralNetwork.dump(new ArrayRealVector(numUnits + 1),
			NeuralNetwork.sigmoid(new ArrayRealVector(z2.getColumnVector(0)))));
		
		BlockRealMatrix z3 = theta2.multiply(act2);
		act3 = NeuralNetwork.convertMatrix(NeuralNetwork.dump(new ArrayRealVector(numUnits + 1),
			NeuralNetwork.sigmoid(new ArrayRealVector(z3.getColumnVector(0)))));
			
		BlockRealMatrix z4 = theta3.multiply(act3);
		return NeuralNetwork.convertMatrix(NeuralNetwork.sigmoid(new ArrayRealVector(z4.getColumnVector(0))));
	}
	
	// Runs back propagation, updates the partial derivative values for one call.
	public void backPropagation(List<Example> examples) {
		BlockRealMatrix delta1 = new BlockRealMatrix(numUnits + 1,numFeatures + 1);
		BlockRealMatrix delta2 = new BlockRealMatrix(numUnits + 1,numUnits + 1);
		BlockRealMatrix delta3 = new BlockRealMatrix(numOutputs,numUnits + 1);
		
		for( Example ex : examples ) {
			forwardPropagation(ex);
		
			BlockRealMatrix error4 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(hypothesis).subtract(ex.y));
	
			ArrayRealVector derivative3 = NeuralNetwork.convertVector(act3).ebeMultiply(NeuralNetwork.convertVector(new BlockRealMatrix(act3
				.scalarMultiply(-1d).scalarAdd(1d).getData())));
			BlockRealMatrix error3 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(theta3.transpose().multiply(error4))
				.ebeMultiply(derivative3));
			
			ArrayRealVector derivative2 = NeuralNetwork.convertVector(act2).ebeMultiply(NeuralNetwork.convertVector(new BlockRealMatrix(act2
				.scalarMultiply(-1d).scalarAdd(1d).getData())));
			BlockRealMatrix error2 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(theta2.transpose().multiply(error3))
				.ebeMultiply(derivative2));
				
			delta1 = delta1.add(error2.multiply(NeuralNetwork.convertMatrix(ex.x).transpose()));
			delta2 = delta2.add(error3.multiply(act2.transpose()));
			delta3 = delta3.add(error4.multiply(act3.transpose()));
		}
		double scalarDelta = 1/(double)this.numExamples;
		
		pDerivative1 = new BlockRealMatrix(delta1.scalarMultiply(scalarDelta).getData());
		pDerivative2 = new BlockRealMatrix(delta2.scalarMultiply(scalarDelta).getData());
		pDerivative3 = new BlockRealMatrix(delta3.scalarMultiply(scalarDelta).getData());
		
		// Regularize the partial derivatives
		RealVector thetaRow1 = this.theta1.getColumnVector(0), thetaRow2 = this.theta2.getColumnVector(0),
		thetaRow3 = this.theta3.getColumnVector(0);
		pDerivative1.setColumnVector(0,new ArrayRealVector(pDerivative1.getColumnVector(0)).add(thetaRow1));
		pDerivative2.setColumnVector(0,new ArrayRealVector(pDerivative2.getColumnVector(0)).add(thetaRow2));
		pDerivative3.setColumnVector(0,new ArrayRealVector(pDerivative3.getColumnVector(0)).add(thetaRow3));
	}
	
	// Sigmoid function that takes an ArrayRealVector.
	public static ArrayRealVector sigmoid( ArrayRealVector arg ) {
		return arg.mapToSelf( new Sigmoid(1e-323,1-(1e-323)) );
	}
	
	// Randomly initialize each of the theta values by [negEpsilon, epsilon] or [-epsilon, epsilon].
	// Can rework the method to initialize more optimally, current naive implementation in quadratic time.
	public static BlockRealMatrix randInitialize( double epsilon, int row, int col ) {
		BlockRealMatrix mat = new BlockRealMatrix(row,col);
		Random r = new Random();
		for( int i = 0; i < row; i++ )
			for( int j = 0; j < col; j++ ) {
				double rand = r.nextDouble() * (2*epsilon) - epsilon;
				mat.addToEntry(i,j,rand);
			}
		return mat;
	}
	
	// Normalize the examples' features.
	public List<Example> meanNormalize( List<Example> exList ) {
		double colorMean = colorSum / colorNum;
		for( Example ex : exList ) {
			ArrayRealVector vec = ex.x;
			for( int i = 0; i < numFeatures + 1; i++ ) {
				vec.setEntry(i,(vec.getEntry(i)-colorMean)/768);
			}
			ex.x = vec;
		}
		return exList;
	}
	
	// Read in the examples' input and output by reading their RGB values and the name (per format), stored in a List.
	public List<Example> readExamples(String path, String set ) {
		int loadedExamples = 0;
		List<Example> ex = new ArrayList<Example>();
		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				loadedExamples++;
				ArrayRealVector input = new ArrayRealVector(numFeatures + 1);
				String filePath = file.toString();
				String subFilePath = filePath.substring(path.length(),filePath.length());
				char letter = subFilePath.charAt(7); // by the format "exampleL0001.jpg".
				BufferedImage bim = ImageIO.read(new File(filePath));
				for( int i = 0; i < 20; i++ ) {
					for( int j = 0; j < 20; j++ ) {
						int colorMean = bim.getRGB(i,j); 
						Color c = new Color(colorMean);
						int sumColor = c.getRed() + c.getGreen() + c.getBlue();
						colorSum += sumColor;
						colorNum++;
						input.addToEntry( (i+1)*(j+1), sumColor); // will normalize using meanNormalize()
					}
				}
				input.addToEntry(0,1); // the x0
				ex.add( new Example(letter, input) );
			}
		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
		if( set.equals( "training" ) )
			numExamples = loadedExamples;
		
		System.out.println( "Loaded in " + loadedExamples + " " + set + " examples." );
		return ex;
	}
	
	// Sums the matrix whose elements have been squared.
	public static double sumSquaredMatrix( BlockRealMatrix mat ) {
		double sum = 0;
		for( int i = 0; i < mat.getRowDimension(); i++ ) {
			for( int j = 0; j < mat.getColumnDimension(); j++ ) {
				sum += Math.pow(mat.getEntry(i,j),2);
			}
		}
		return sum;
	}
	
	// Sums the given vector, returns a double value.
	public static double sumVector( ArrayRealVector vec ) {
		double sum = 0;
		for( int i = 0; i < vec.getDimension(); i++ )
			sum += vec.getEntry(i);
		return sum;
	}
	
	// Print the hypothesis - take the highest value across all the entries and convert to the character.
	// Assumes hypothesis is a 1 x n matrix.
	public static void printHypothesis( BlockRealMatrix hyp ) {
		double max = -1;
		int index = 0; // default
		double[] data = hyp.getColumnVector(0).toArray();
		for( int i = 0; i < data.length; i++ ) {
			if( data[i] > max ) {
				max = data[i];
				index = i;
			}
		}
		System.out.println( "The predicted character is: " + (char) (index+65) );
	}
	
	// Auxiliary method to turn an ArrayRealVector to a BlockRealMatrix.
	public static BlockRealMatrix convertMatrix( ArrayRealVector vec ) {
		BlockRealMatrix mat = new BlockRealMatrix(vec.getDimension(),1);
		for( int i = 0; i < vec.getDimension(); i++ )
			mat.setEntry( i, 0, vec.getEntry(i));
		return mat;
	}

	// Auxiliary method to turn a BlockRealMatrix to an ArrayRealVector - works only if one dimension of the matrix = 1.
	public static ArrayRealVector convertVector( BlockRealMatrix mat ) {
		if( mat.getRowVector(0).getDimension() == 1 )
			return new ArrayRealVector(mat.getColumnVector(0));
		else
			return new ArrayRealVector(mat.getRowVector(0));
	}
	
	// Auxiliary method to dump a resultant array into an existing one considering the dummy 0th element = 1.
	public static ArrayRealVector dump( ArrayRealVector finalVec, ArrayRealVector dumpVec ) {
		finalVec.setEntry(0,1);
		for( int i = 1; i < dumpVec.getDimension(); i++ )
			finalVec.setEntry(i,dumpVec.getEntry(i));
		return finalVec;
	}
}
