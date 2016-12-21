
include_directories(
  model
)

add_executable(NgramLabeler NgramLabeler.cpp)
target_link_libraries(NgramLabeler ${LIBS})
