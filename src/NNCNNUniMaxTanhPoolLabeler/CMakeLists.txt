
include_directories(
  model
)

add_executable(NNCNNUniMaxTanhPoolLabeler NNCNNUniMaxTanhPoolLabeler.cpp)
target_link_libraries(NNCNNUniMaxTanhPoolLabeler ${LIBS})
