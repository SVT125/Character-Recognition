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
			dir = Paths.get("C:/Users/James/Programming/CharacterRecognition/dumping folder/" + args[1]);
		else 
			dir = Paths.get("C:/Users/James/Programming/CharacterRecognition/dumping folder/");
		
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				BufferedImage bim = ImageIO.read(new File(file.toString()));
				Image img = bim.getScaledInstance(20,20,bim.SCALE_SMOOTH);
				BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);
				Graphics2D bGr = bimage.createGraphics();
				bGr.drawImage(img, 0, 0, null);
				bGr.dispose();
				File output = new File("example" + args[0] + Integer.toString(counter++));
				ImageIO.write(bimage, "PNG", output);
			}

		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
		
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				File currentFile = new File( file.toString() );
				File newFile = new File( file.toString() + ".png" );
				currentFile.renameTo(newFile);
			}

		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
	}
}