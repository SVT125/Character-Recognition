// Auxiliary methods for the matrices and vectors in the Apache Commons Math 3.1 API.

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.analysis.function.Sigmoid;
import java.io.*;

class MatrixAuxiliaryMethods {
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
}