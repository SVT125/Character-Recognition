// Will expect images of dimensions 30 x 30 pixels (900 features)as input.
// Use file naming format e.g. "exampleL0001.jpg", where we read in the file name and take the 8th character as the output y, here "L".

import org.apache.commons.math3.linear.*;
import java.nio.file.*;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

public class NeuralNetwork {
	public static void main( String[] args ) {
	
	}
	public static List<Example> readExamples( String fileName ) throws IOException {
		List<Example> ex = new ArrayList<Example>();
		Path dir = Paths.get("/examples" );
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				ArrayRealVector input = new ArrayRealVector(900);
				String filePath = file.toString();
				char letter = filePath.charAt(7); // by the format "exampleL0001.jpg".
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
