# Locate SDL2 library
FIND_LIBRARY(SDL_LIBRARY SDL2
	$ENV{SDLDIR}/lib
	/usr/lib
	/usr/local/lib
)

FIND_PATH(SDL_INCLUDE_DIR SDL.h
	$ENV{SDLDIR}/include/SDL2
	/usr/include/SDL2
	/usr/local/include/SDL2
)

SET( SDL_FOUND "NO" )
IF(SDL_LIBRARY)
	SET( SDL_FOUND "YES" )
	mark_as_advanced(SDL_INCLUDE_DIR SDL_LIBRARY)
ENDIF(SDL_LIBRARY)