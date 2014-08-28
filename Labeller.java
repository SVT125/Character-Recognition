public class Preprocessor {
	public static void main( String[] args ) {
		Path dir;
		if( args.length > 1 )
			dir = Paths.get("/examples" + args[1]);
		else 
			System.out.println( "Provide the folder extension for files to rename as the second argument." );
		
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				Files.move(file, file.resolveSibling("newname"));
			}
		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
	}
}