import java.nio.file.*;
import java.io.*;
import java.util.*;
import java.awt.image.*;
import java.awt.Graphics2D;
import java.awt.Image;
import javax.imageio.ImageIO;

// Auxiliary class for renaming images to the appropriate format and rescaling them to the 30x30 image size.
public class Preprocessor {
	public static void main( String[] args ) {
		int counter = 0;
		Path dir;
		if( args.length > 1 )
			dir = Paths.get("C:/Users/James/Programming/examples/" + args[1]);
		else 
			dir = Paths.get("/Programming/examples");
		
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