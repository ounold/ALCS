#include <vector>

#include <Maze/Mazes/MazeF4.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    MazeF4::MazeF4() : IMazeEnvironment(
    7 * 4,
    {
        .mazeWidth = 7,
        .mazeHeight = 6,
        .goalState = {1, 5},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, PRIZE,    OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 0, 2, 1},
                    //            {1, 0, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 0, 1, 1},
                    //            {1, 0, 1, 1, 1, 1, 1},
                    //            {1, 1, 1, 1, 1, 1, 1},
            }
    }) {}
}