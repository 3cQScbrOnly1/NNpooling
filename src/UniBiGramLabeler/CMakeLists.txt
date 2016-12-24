
include_directories(
  model
)

add_executable(UniBiGramLabeler UniBiGramLabeler.cpp)
target_link_libraries(UniBiGramLabeler ${LIBS})
