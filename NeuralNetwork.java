// Will expect images of dimensions 30 x 30 pixels (900 features)as input.
// Use file naming format e.g. "exampleL0001.jpg", where we read in the file name and take the 8th character as the output y, here "L".

// Initial neural network implementation will have 100 units per layer, 2 hidden layers.

import org.apache.commons.math3.linear.*;
import java.nio.file.*;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

public class NeuralNetwork {
	static List<Example> examples;
	static BlockRealMatrix theta1;
	static BlockRealMatrix theta2;
	static ArrayRealVector act1 = new ArrayRealVector(100);
	static ArrayRealVector act2 = new ArrayRealVector(100);
	
	public static void main( String[] args ) {
		NeuralNetwork.examples = NeuralNetwork.readExamples("C:/Users/James/Programming/examples/");
		System.out.println( "Loaded in examples." );
		theta1 = NeuralNetwork.randInitialize( 2, 100, 101 );
		theta2 = NeuralNetwork.randInitialize( 2, 100, 101 );
		
	}
	
	// Randomly initialize each of the theta values by [negEpsilon, epsilon] or [-epsilon, epsilon].
	// Maybe rework the method to initialize more optimally, current naive implementation in quadratic time.
	public static BlockRealMatrix randInitialize( int epsilon, int row, int col ) {
		BlockRealMatrix mat = new BlockRealMatrix(row,col);
		Random r = new Random();
		for( int i = 0; i < row; i++ )
			for( int j = 0; j < col; j++ ) {
				int rand = r.nextInt(epsilon) - 2;
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
				ArrayRealVector input = new ArrayRealVector(900);
				String filePath = file.toString();
				String subFilePath = filePath.substring(path.length(),filePath.length());
				char letter = subFilePath.charAt(7); // by the format "exampleL0001.jpg".
				BufferedImage bim = ImageIO.read(new File(filePath));
				for( int i = 0; i < 30; i++ ) {
					for( int j = 0; j < 30; j++ ) {
						input.addToEntry( (i+1)*(j+1)-1, bim.getRGB(i,j));
					}
				}
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
	
}
