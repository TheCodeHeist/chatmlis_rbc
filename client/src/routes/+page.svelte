<script lang="ts">
	import { Input } from '$lib/components/ui/input/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { SendHorizontal } from 'lucide-svelte';
</script>

<main class="h-screen w-full bg-neutral-900">
	<header class="flex items-center justify-center py-4 fixed top-0 w-full">
		<p class="text-xl font-semibold text-emerald-500">ChatMLIS</p>
	</header>

	<article class="h-full w-full flex items-center justify-center flex-col gap-8">
		<p class="text-white text-2xl">How can I assist you today?</p>

		<div class="flex items-center justify-center gap-2 w-full">
			<Input type="text" placeholder="Ask anything" class="w-1/4" />
			<Button size="icon" class="size-8 cursor-pointer">
				<SendHorizontal />
			</Button>
		</div>
	</article>

	<footer class="text-center text-white py-4 fixed w-full bottom-0">
		<p>Made with ❤️ by MLIS Robotics Club</p>
	</footer>
</main>
