import org.apache.commons.math3.linear.*;

public class NeuralNetwork {
	public static void main( String[] args ) {
	
	}
	public static List<Example> readExamples( String fileName ) {
		
	}
}

class Example {
	double[][] mat;
	ArrayRealVector y = new ArrayVector(26);
	int type; // 0 (negative) or 1 (positive)
	
	// Construct an Example given the letter the image represents and the inputs. 
	// TODO : inputs
	public Example( char letter ) {
		int num = (int) letter - 65; // Index of the 0-indexed vector to store in, takes A - Z (will only deal with upper case).
		y.setValue(num,1.0); // Set the example
	}
}