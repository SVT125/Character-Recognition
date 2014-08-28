import java.nio.file.*;
import java.io.*;
import java.util.*;
import java.awt.image.*;
import java.awt.Graphics2D;
import java.awt.Image;
import javax.imageio.ImageIO;

public class Preprocessor {
	public static void main( String[] args ) throws IllegalArgumentException {
		int counter = 0;
		Path dir;
		if( args.length > 1 )
			dir = Paths.get("/examples" + args[1]);
		else 
			throw new IllegalArgumentException( "Provide the folder extension for files to rename as the second argument." );
		
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				BufferedImage bim = ImageIO.read(new File(file.toString()));
				Image img = bim.getScaledInstance(30,30,bim.SCALE_SMOOTH);
				BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);
				Graphics2D bGr = bimage.createGraphics();
				bGr.drawImage(img, 0, 0, null);
				bGr.dispose();
				File output = new File("example" + args[0] + Integer.toString(counter++));
				ImageIO.write(bimage, "png", output);
			}
		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
	}
}