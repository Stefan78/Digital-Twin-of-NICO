using Codice.CM.Common.Tree;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(MoveWithSliders))]
public class JointSliderControllerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        var controller = (MoveWithSliders)target;

        // Make sure both lists are the same length
        while (controller.jointTargets.Count < controller.activeJoints.Count)
            controller.jointTargets.Add(0f);

        while (controller.jointTargets.Count > controller.activeJoints.Count)
            controller.jointTargets.RemoveAt(controller.jointTargets.Count - 1);

        for (int i = 0; i < controller.activeJoints.Count; i++)
        {
            var joint = controller.activeJoints[i];
            if (joint == null)
            {
                EditorGUILayout.LabelField($"Joint {i}", "Missing reference");
                continue;
            }

            var drive = joint.xDrive;
            float min = drive.lowerLimit;
            float max = drive.upperLimit;

            EditorGUILayout.LabelField(joint.name);
            controller.jointTargets[i] = EditorGUILayout.Slider(controller.jointTargets[i], min, max);
        }

        // Save changes
        if (GUI.changed)
        {
            EditorUtility.SetDirty(controller);
        }
    }
}
