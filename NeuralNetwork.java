// Will expect images of dimensions 30 x 30 pixels (900 features)as input.
// Use file naming format e.g. "exampleL0001.jpg", where we read in the file name and take the 8th character as the output y, here "L".

// Initial neural network implementation will have 100 units per layer, 2 hidden layers.

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.analysis.function.Sigmoid;
import java.nio.file.*;
import java.io.*;
import java.util.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class NeuralNetwork {
	static List<Example> examples;
	static BlockRealMatrix theta1;
	static BlockRealMatrix theta2;
	static BlockRealMatrix theta3;
	static ArrayRealVector hypothesis = new ArrayRealVector(26); // layer 4, final vector result
	
	public static void main( String[] args ) {
		NeuralNetwork.examples = NeuralNetwork.readExamples("C:/Users/James/Programming/examples/");
		System.out.println( "Loaded in examples." );
		theta1 = NeuralNetwork.randInitialize( 2, 100, 901 );
		theta2 = NeuralNetwork.randInitialize( 2, 100, 101 );
		theta3 = NeuralNetwork.randInitialize( 2, 26, 101 );
		
		// Intermediary steps...
		
		hypothesis = NeuralNetwork.forwardPropagation(NeuralNetwork.examples.get(1));
	}
	
	// Runs forward propagation, given this particular neural network.
	// Can readjust to take arguments of number of units and layers.
	public static ArrayRealVector forwardPropagation(Example ex) {
		BlockRealMatrix z2 = NeuralNetwork.theta1.multiply(NeuralNetwork.convertMatrix(ex.x));
		BlockRealMatrix act2 = NeuralNetwork.convertMatrix(NeuralNetwork.dump(new ArrayRealVector(101),
			NeuralNetwork.sigmoid(new ArrayRealVector(z2.transpose().getColumnVector(0)))));
		return new ArrayRealVector(); // placeholder...
	}
	
	// Sigmoid function that takes an ArrayRealVector.
	// Rework the method to use on matrix, requires matrix exponentiation.
	public static ArrayRealVector sigmoid( ArrayRealVector arg ) {
		return arg.mapToSelf( new Sigmoid() );
	}
	
	// Randomly initialize each of the theta values by [negEpsilon, epsilon] or [-epsilon, epsilon].
	// Can rework the method to initialize more optimally, current naive implementation in quadratic time.
	public static BlockRealMatrix randInitialize( int epsilon, int row, int col ) {
		BlockRealMatrix mat = new BlockRealMatrix(row,col);
		Random r = new Random();
		for( int i = 0; i < row; i++ )
			for( int j = 0; j < col; j++ ) {
				int rand = r.nextInt(epsilon) - epsilon;
				mat.addToEntry(i,j,(double)rand);
			}
		return mat;
	}
	
	// Read in the examples' input and output by reading their RGB values and the name (per format), stored in a List.
	public static List<Example> readExamples(String path ) {
		List<Example> ex = new ArrayList<Example>();
		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				ArrayRealVector input = new ArrayRealVector(901);
				String filePath = file.toString();
				String subFilePath = filePath.substring(path.length(),filePath.length());
				char letter = subFilePath.charAt(7); // by the format "exampleL0001.jpg".
				BufferedImage bim = ImageIO.read(new File(filePath));
				for( int i = 0; i < 30; i++ ) {
					for( int j = 0; j < 30; j++ ) {
						input.addToEntry( (i+1)*(j+1), bim.getRGB(i,j));
					}
				}
				input.addToEntry(1,1); // the x0
				ex.add( new Example(letter, input) );
			}
		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
		return ex;
	}
	
	// Prints the matrix by creating an 2d double array primitive to loop over, works for non n x n matrices.
	// Auxiliary method to check proper vector accesses.
	public static void printMatrix( BlockRealMatrix brm ) {
		double[][] mat = brm.getData();
		for( int i = 0; i < mat.length; i++ ) {
			for( int j = 0; j < mat[i].length; j++ ) {
				System.out.print( mat[i][j] + " " );
			}
			System.out.println();
		}
	}
	
	// Auxiliary method to turn an ArrayRealVector to a BlockRealMatrix.
	public static BlockRealMatrix convertMatrix( ArrayRealVector vec ) {
		BlockRealMatrix mat = new BlockRealMatrix(vec.getDimension(),1);
		for( int i = 0; i < vec.getDimension(); i++ )
			mat.setEntry( i, 0, vec.getEntry(i));
		return mat;
	}

	// Auxiliary method to dump a resultant array into an existing one considering the dummy 0th element = 1.
	public static ArrayRealVector dump( ArrayRealVector finalVec, ArrayRealVector dumpVec ) {
		finalVec.setEntry(0,1);
		for( int i = 0; i < dumpVec.getDimension(); i++ )
			finalVec.setEntry(i+1,dumpVec.getEntry(i));
		return finalVec;
	}
}

