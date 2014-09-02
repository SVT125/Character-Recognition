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
	static List<Example> trainingExamples;
	static List<Example> testExamples;
	
	static BlockRealMatrix theta1;
	static BlockRealMatrix theta2;
	static BlockRealMatrix theta3;
	static BlockRealMatrix pDerivative1;
	static BlockRealMatrix pDerivative2;
	static BlockRealMatrix pDerivative3;
	static BlockRealMatrix act2;
	static BlockRealMatrix act3;
	
	static BlockRealMatrix hypothesis = new BlockRealMatrix(26,1); // layer 4, final vector result
	static BlockRealMatrix testHypothesis = new BlockRealMatrix(26,1); // the test hypothesis when calculating the cost
	
	static double learningRate = .01; // rate multiplied by the partial derivative. Greater values mean faster convergence but possible divergence
	static double regularizationRate = .01; // regularization rate 
	static double colorSum = 0; // sum of all numbers r, g, b across all examples
	static double colorNum = 0; // number of pixels across all examples
	static int numExamples = 0; // number of examples
	
	public static void main( String[] args ) throws Exception {
		NeuralNetwork.trainingExamples = NeuralNetwork.readExamples("C:/Users/James/Programming/CharacterRecognition/trainingexamples/", "training");
		NeuralNetwork.testExamples = NeuralNetwork.readExamples("C:/Users/James/Programming/CharacterRecognition/testexamples/", "test");
		NeuralNetwork.trainingExamples = NeuralNetwork.meanNormalize( trainingExamples );
		NeuralNetwork.testExamples = NeuralNetwork.meanNormalize( testExamples );

		
		theta1 = NeuralNetwork.randInitialize( 1, 101, 401 );
		System.out.println( "Randomly initialized first parameter vector." );
		theta2 = NeuralNetwork.randInitialize( 1, 101, 101 );
		System.out.println( "Randomly initialized second parameter vector." );
		theta3 = NeuralNetwork.randInitialize( 1, 26, 101 );
		System.out.println( "Randomly initialized third parameter vector." );
		System.out.println("---------------------");
		
		NeuralNetwork.train(Integer.parseInt(args[0])); // Can adjust to iterate until convergence instead
		
		hypothesis = NeuralNetwork.forwardPropagation( NeuralNetwork.testExamples.get(0));
		System.out.println("---------------------");
		NeuralNetwork.printHypothesis(hypothesis);
		//NeuralNetwork.printMatrix(hypothesis);
	}
	
	// Runs gradient descent until the cost is less than some epsilon.
	public static void train(double epsilon) {
		int iteration = 1;
		while( NeuralNetwork.calculateCost() > epsilon ) {
			NeuralNetwork.gradientDescent(1);
			System.out.println( "Cost of iteration " + iteration++ + " is: " + NeuralNetwork.calculateCost());
		}
	}
	
	// Calculates the cost function.
	public static double calculateCost() {
		double cost = 0;
		double threshold = Math.pow(Math.E,-323); // pre-calculated value for loops
		// sum over all examples
		for( int i = 0; i < NeuralNetwork.numExamples; i++ ) { 
			NeuralNetwork.testHypothesis = NeuralNetwork.forwardPropagation(NeuralNetwork.trainingExamples.get(i));
			// sum over all elements of the hypothesis
			for( int j = 0; j < NeuralNetwork.hypothesis.getRowDimension(); j++ ) {
				double firstLogArgument = NeuralNetwork.testHypothesis.getEntry(j,0), secondLogArgument = 1 - NeuralNetwork.testHypothesis.getEntry(j,0);
				if( firstLogArgument < threshold )
					firstLogArgument = threshold;
				if( secondLogArgument < threshold )
					secondLogArgument = threshold;	
				cost += ( (NeuralNetwork.trainingExamples.get(i).y.getEntry(j) * Math.log(firstLogArgument)) + 
					((1 - NeuralNetwork.trainingExamples.get(i).y.getEntry(j)) * Math.log(secondLogArgument)) ); // define the hypothesis for each example
				// regularization term
			}
		}
		cost = ((double)(-1)/(double) (NeuralNetwork.numExamples)) * cost;
		cost += (NeuralNetwork.regularizationRate/(2*NeuralNetwork.numExamples)) * (NeuralNetwork.sumSquaredMatrix(NeuralNetwork.theta1) + 
			NeuralNetwork.sumSquaredMatrix(NeuralNetwork.theta2) + NeuralNetwork.sumSquaredMatrix(NeuralNetwork.theta3));
		return cost;
	}
	
	// Runs numIterations iterations of batch gradient descent using the class' partial derivative terms calculated from backprop and the learning rate.
	public static void gradientDescent( int numIterations ) {
		for( int i = 0; i < numIterations; i++ ) {
			for( Example ex : NeuralNetwork.trainingExamples ) 
				NeuralNetwork.hypothesis = NeuralNetwork.forwardPropagation( ex );
			NeuralNetwork.backPropagation( NeuralNetwork.trainingExamples ); // updates partial derivatives
			NeuralNetwork.theta1 = NeuralNetwork.theta1.subtract( NeuralNetwork.pDerivative1.scalarMultiply(NeuralNetwork.learningRate) );
			NeuralNetwork.theta2 = NeuralNetwork.theta2.subtract( NeuralNetwork.pDerivative2.scalarMultiply(NeuralNetwork.learningRate) );
			NeuralNetwork.theta3 = NeuralNetwork.theta3.subtract( NeuralNetwork.pDerivative3.scalarMultiply(NeuralNetwork.learningRate) );
		}
	}
	
	// Runs forward propagation, given this particular neural network.
	// Can readjust to take arguments of number of units and layers.
	public static BlockRealMatrix forwardPropagation(Example ex) {
		BlockRealMatrix z2 = NeuralNetwork.theta1.multiply(NeuralNetwork.convertMatrix(ex.x));
		NeuralNetwork.act2 = NeuralNetwork.convertMatrix(NeuralNetwork.dump(new ArrayRealVector(101),
			NeuralNetwork.sigmoid(new ArrayRealVector(z2.getColumnVector(0)))));
		
		BlockRealMatrix z3 = NeuralNetwork.theta2.multiply(act2);
		NeuralNetwork.act3 = NeuralNetwork.convertMatrix(NeuralNetwork.dump(new ArrayRealVector(101),
			NeuralNetwork.sigmoid(new ArrayRealVector(z3.getColumnVector(0)))));
			
		BlockRealMatrix z4 = NeuralNetwork.theta3.multiply(act3);
		return NeuralNetwork.convertMatrix(NeuralNetwork.sigmoid(new ArrayRealVector(z4.getColumnVector(0))));
	}
	
	// Runs back propagation, updates the partial derivative values for one call.
	public static void backPropagation(List<Example> examples) {
		BlockRealMatrix delta1 = new BlockRealMatrix(101,401);
		BlockRealMatrix delta2 = new BlockRealMatrix(101,101);
		BlockRealMatrix delta3 = new BlockRealMatrix(26,101);
		
		for( Example ex : examples ) {
			NeuralNetwork.forwardPropagation(ex);
		
			BlockRealMatrix error4 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(NeuralNetwork.hypothesis).subtract(ex.y));
	
			ArrayRealVector derivative3 = NeuralNetwork.convertVector(NeuralNetwork.act3).ebeMultiply(NeuralNetwork.convertVector(new BlockRealMatrix(NeuralNetwork.act3
				.scalarMultiply(-1d).scalarAdd(1d).getData())));
			BlockRealMatrix error3 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(NeuralNetwork.theta3.transpose().multiply(error4))
				.ebeMultiply(derivative3));
			
			ArrayRealVector derivative2 = NeuralNetwork.convertVector(NeuralNetwork.act2).ebeMultiply(NeuralNetwork.convertVector(new BlockRealMatrix(NeuralNetwork.act2
				.scalarMultiply(-1d).scalarAdd(1d).getData())));
			BlockRealMatrix error2 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(NeuralNetwork.theta2.transpose().multiply(error3))
				.ebeMultiply(derivative2));
				
			delta1 = delta1.add(error2.multiply(NeuralNetwork.convertMatrix(ex.x).transpose()));
			delta2 = delta2.add(error3.multiply(NeuralNetwork.act2.transpose()));
			delta3 = delta3.add(error4.multiply(NeuralNetwork.act3.transpose()));
		}
		double scalarDelta = 1/(double)NeuralNetwork.numExamples;
		
		NeuralNetwork.pDerivative1 = new BlockRealMatrix(delta1.scalarMultiply(scalarDelta).getData());
		NeuralNetwork.pDerivative2 = new BlockRealMatrix(delta2.scalarMultiply(scalarDelta).getData());
		NeuralNetwork.pDerivative3 = new BlockRealMatrix(delta3.scalarMultiply(scalarDelta).getData());
		
		// Regularize the partial derivatives
		RealVector thetaRow1 = NeuralNetwork.theta1.getColumnVector(0), thetaRow2 = NeuralNetwork.theta2.getColumnVector(0),
			thetaRow3 = NeuralNetwork.theta3.getColumnVector(0);
		NeuralNetwork.pDerivative1.setColumnVector(0,new ArrayRealVector(NeuralNetwork.pDerivative1.getColumnVector(0)).add(thetaRow1));
		NeuralNetwork.pDerivative2.setColumnVector(0,new ArrayRealVector(NeuralNetwork.pDerivative2.getColumnVector(0)).add(thetaRow2));
		NeuralNetwork.pDerivative3.setColumnVector(0,new ArrayRealVector(NeuralNetwork.pDerivative3.getColumnVector(0)).add(thetaRow3));
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
	public static List<Example> meanNormalize( List<Example> exList ) {
		double colorMean = NeuralNetwork.colorSum / NeuralNetwork.colorNum;
		for( Example ex : exList ) {
			ArrayRealVector vec = ex.x;
			for( int i = 0; i < 401; i++ ) {
				vec.setEntry(i,(vec.getEntry(i)-colorMean)/768);
			}
			ex.x = vec;
		}
		return exList;
	}
	
	// Read in the examples' input and output by reading their RGB values and the name (per format), stored in a List.
	public static List<Example> readExamples(String path, String set ) {
		int loadedExamples = 0;
		List<Example> ex = new ArrayList<Example>();
		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				loadedExamples++;
				ArrayRealVector input = new ArrayRealVector(401);
				String filePath = file.toString();
				String subFilePath = filePath.substring(path.length(),filePath.length());
				char letter = subFilePath.charAt(7); // by the format "exampleL0001.jpg".
				BufferedImage bim = ImageIO.read(new File(filePath));
				for( int i = 0; i < 20; i++ ) {
					for( int j = 0; j < 20; j++ ) {
						int colorMean = bim.getRGB(i,j); 
						Color c = new Color(colorMean);
						int sumColor = c.getRed() + c.getGreen() + c.getBlue();
						NeuralNetwork.colorSum += sumColor;
						NeuralNetwork.colorNum++;
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
			NeuralNetwork.numExamples = loadedExamples;
		
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
	
	// Auxiliary method to check proper vector accesses.
	// Prints the matrix by creating an 2d double array primitive to loop over, works for non n x n matrices.
	public static void printMatrix( BlockRealMatrix brm ) {
		double[][] mat = brm.getData();
		for( int i = 0; i < mat.length; i++ ) {
			for( int j = 0; j < mat[i].length; j++ ) {
				System.out.print( mat[i][j] + " " );
			}
			System.out.println();
		}
	}
	
	// Auxiliary method to print the vector.
	public static void printVector( ArrayRealVector vec ) {
		double[] array = vec.toArray();
		for( int i = 0; i < array.length; i++ ) {
			System.out.println( array[i] );
		}
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

	// Auxiliary method to check the dimensions of the input matrix.
	public static void printDimensions( BlockRealMatrix mat ) {
		System.out.println( "Number of rows: " + mat.getColumnVector(0).getDimension() );
		System.out.println( "Number of columns: " + mat.getRowVector(0).getDimension() );		
	}
	
	// Auxiliary method to better examine the matrix's values.
	public static void printMatrixToFile( BlockRealMatrix mat ) throws Exception {
		PrintWriter writer = new PrintWriter("matrixdump.txt", "UTF-8");
		double[][] matArray = mat.getData();
		for( int i = 0; i < matArray.length; i++ ) {
			for( int j = 0; j < matArray[i].length; j++ ) {
				writer.print( matArray[i][j] + " " );
			}
			writer.println();
		}
	}
}