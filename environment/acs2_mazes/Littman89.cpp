#include <vector>

#include <Maze/Mazes/Littman89.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"



namespace Environments
{
    using enum EFieldState;

    Littman89::Littman89() : IMazeEnvironment(
    9 * 4,
    {
        .mazeWidth = 9,
        .mazeHeight = 7,
        .goalState = {4, 7},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, PRIZE,    OBSTACLE },
                    { OBSTACLE, OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 1, 0, 0, 0, 0, 0, 1, 1},
                    //            {1, 0, 0, 1, 0, 1, 0, 0, 1},
                    //            {1, 1, 0, 1, 0, 1, 0, 1, 1},
                    //            {1, 0, 0, 1, 0, 1, 0, 2, 1},
                    //            {1, 1, 0, 0, 0, 0, 0, 1, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1}
            },
    }) {}
}