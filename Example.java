import org.apache.commons.math3.linear.*;

class Example {
	ArrayRealVector y = new ArrayRealVector(26);
	ArrayRealVector x;

	
	// Construct an Example given the letter the image represents and the inputs. 
	// TODO : inputs
	public Example( char letter, ArrayRealVector input ) {
		int num = (int) letter - 65; // Index of the 0-indexed vector to store in, assumes file name read in is lower case.
		y.setEntry(num,1.0); // Set the example
		x = input;
	}
}