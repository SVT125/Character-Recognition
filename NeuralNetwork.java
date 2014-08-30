// Will expect images of dimensions 20 x 20 pixels (400 features)as input.
// Use file naming format e.g. "exampleL0001.jpg", where we read in the file name and take the 8th character as the output y, here "L".

// Initial neural network implementation will have 100 units per layer, 2 hidden layers.

/* To-do: serialize the results, although running time for current >200 examples is only a few seconds.
   Partial derivatives
   Implement gradient descent or some other advanced algorithm
   Regularization
*/ 

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
	static double colorSum = 0; // sum of all numbers r, g, b across all examples
	static double colorNum = 0; // number of pixels across all examples
	static int numExamples = 0; // number of examples
	
	public static void main( String[] args ) {
		NeuralNetwork.trainingExamples = NeuralNetwork.readExamples("C:/Users/James/Programming/CharacterRecognition/trainingexamples/", "training");
		NeuralNetwork.testExamples = NeuralNetwork.readExamples("C:/Users/James/Programming/CharacterRecognition/testexamples/", "test");
		NeuralNetwork.trainingExamples = NeuralNetwork.meanNormalize( NeuralNetwork.trainingExamples );
		NeuralNetwork.trainingExamples = NeuralNetwork.meanNormalize( NeuralNetwork.testExamples );
		
		theta1 = NeuralNetwork.randInitialize( 1, 101, 401 );
		System.out.println( "Randomly initialized first parameter vector." );
		theta2 = NeuralNetwork.randInitialize( 1, 101, 101 );
		System.out.println( "Randomly initialized second parameter vector." );
		theta3 = NeuralNetwork.randInitialize( 1, 26, 101 );
		System.out.println( "Randomly initialized third parameter vector." );
		
		/* 
		Intermediary steps... (backprop, optimizing theta)
		*/
		NeuralNetwork.forwardPropagation(NeuralNetwork.trainingExamples.get(0)); // Testing the forward prop method with 1 example
		NeuralNetwork.backPropagation(NeuralNetwork.trainingExamples.get(0)); // Testing the backward prop method with 1 example
		
		System.out.println("---------------------");
		NeuralNetwork.printHypothesis(hypothesis);
	}
	
	// Runs forward propagation, given this particular neural network.
	// Can readjust to take arguments of number of units and layers.
	public static void forwardPropagation(Example ex) {
		BlockRealMatrix z2 = NeuralNetwork.theta1.multiply(NeuralNetwork.convertMatrix(ex.x));
		NeuralNetwork.act2 = NeuralNetwork.convertMatrix(NeuralNetwork.dump(new ArrayRealVector(101),
			NeuralNetwork.sigmoid(new ArrayRealVector(z2.getColumnVector(0)))));
		
		BlockRealMatrix z3 = NeuralNetwork.theta2.multiply(act2);
		NeuralNetwork.act3 = NeuralNetwork.convertMatrix(NeuralNetwork.dump(new ArrayRealVector(101),
			NeuralNetwork.sigmoid(new ArrayRealVector(z3.getColumnVector(0)))));
			
		BlockRealMatrix z4 = NeuralNetwork.theta3.multiply(act3);
		NeuralNetwork.hypothesis = NeuralNetwork.convertMatrix(NeuralNetwork.sigmoid(new ArrayRealVector(z4.getColumnVector(0))));
	}
	
	// Runs back propagation, updates the partial derivative values for one call.
	public static void backPropagation(Example ex) {
		BlockRealMatrix error4 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(NeuralNetwork.hypothesis).subtract(ex.y));

		ArrayRealVector derivative3 = NeuralNetwork.convertVector(NeuralNetwork.act3).ebeMultiply(NeuralNetwork.convertVector(new BlockRealMatrix(NeuralNetwork.act3
			.scalarMultiply(-1d).scalarAdd(1d).getData())));
		BlockRealMatrix error3 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(NeuralNetwork.theta3.transpose().multiply(error4))
			.ebeMultiply(derivative3));
			
		ArrayRealVector derivative2 = NeuralNetwork.convertVector(NeuralNetwork.act2).ebeMultiply(NeuralNetwork.convertVector(new BlockRealMatrix(NeuralNetwork.act2
			.scalarMultiply(-1d).scalarAdd(1d).getData())));
		BlockRealMatrix error2 = NeuralNetwork.convertMatrix(NeuralNetwork.convertVector(NeuralNetwork.theta2.transpose().multiply(error3))
			.ebeMultiply(derivative2));
			
		BlockRealMatrix delta1 = error2.multiply(NeuralNetwork.convertMatrix(ex.x).transpose());
		BlockRealMatrix delta2 = error3.multiply(NeuralNetwork.act2.transpose());
		BlockRealMatrix delta3 = error4.multiply(NeuralNetwork.act3.transpose());
		
		NeuralNetwork.pDerivative1 = delta1.scalarMultiply((double) (1/NeuralNetwork.numExamples));
		NeuralNetwork.pDerivative2 = delta2.scalarMultiply((double) (1/NeuralNetwork.numExamples));
		NeuralNetwork.pDerivative3 = delta3.scalarMultiply((double) (1/NeuralNetwork.numExamples));
	}
	
	// Sigmoid function that takes an ArrayRealVector.
	// Rework the method to use on matrix, requires matrix exponentiation.
	public static ArrayRealVector sigmoid( ArrayRealVector arg ) {
		return arg.mapToSelf( new Sigmoid() );
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
		
		System.out.println( "Loaded in " + numExamples + " examples." );
		return ex;
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
}