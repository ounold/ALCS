#include <vector>

#include <Maze/Mazes/MazeF3.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    MazeF3::MazeF3() : IMazeEnvironment(
    6 * 4,
    {
        .mazeWidth = 6,
        .mazeHeight = 6,
        .goalState = {1, 4},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, PRIZE,    OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 2, 1},
                    //            {1, 0, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 1, 1},
                    //            {1, 0, 1, 1, 1, 1},
                    //            {1, 1, 1, 1, 1, 1},
            },
    }) {}
}