
import robot_franka_emika.franka_emika as franka_emika
import robot_baxter.baxter as baxter
import robot_sice.sice as sice
import robot_sice_ex.sice_ex as sice_ex
import robot_particle.particle as particle
import robot_yamanaka.yamanaka as yamanaka

def get_robot_model(robot_name):
    if robot_name == "baxter":
        return baxter
    elif robot_name == "franka_emika":
        return franka_emika
    elif robot_name == "sice":
        return sice
    elif robot_name == "sice_ex":
        return sice_ex
    elif robot_name == "particle":
        return particle
    elif robot_name == "yamanaka":
        return yamanaka
    else:
        assert False


