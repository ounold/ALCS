#include <vector>

#include <Maze/Mazes/MazeE3.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments
{
    using enum EFieldState;

    MazeE3::MazeE3() : IMazeEnvironment(
    11 * 4,
    {
            .mazeWidth = 11,
            .mazeHeight = 11,
            .goalState = {5, 5},
            .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, PRIZE,    CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
            },

    }) {}
}