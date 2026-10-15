import type { LiteCourse } from "../../lib/course.ts";
import { MeView } from "./MeView.tsx";
import { PathView } from "./PathView.tsx";
import { ReviewView } from "./ReviewView.tsx";
import { ScheduleView } from "./ScheduleView.tsx";
import { Shell, type Tab } from "./Shell.tsx";

export default function Home({ course, tab }: { course: LiteCourse; tab: Tab }) {
  const View = { path: PathView, schedule: ScheduleView, review: ReviewView, me: MeView }[tab];
  return <Shell course={course} tab={tab}><View course={course} /></Shell>;
}
